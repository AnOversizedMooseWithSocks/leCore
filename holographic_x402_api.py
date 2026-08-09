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
import base64
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
import struct
from string import Template
import subprocess
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlsplit
import zlib

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
_MEMORY_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
MAX_QUERY_CHARS = 8192
MAX_TASK_CHARS = 8192
MAX_MEMORY_CHARS = 65536
MAX_MEMORY_LABEL_CHARS = 256
MAX_MEMORY_METADATA_BYTES = 16384
MAX_RECALL_K = 100
MEMORY_BACKEND_CORE = "core"
MEMORY_BACKEND_NOSQLITE = "nosqlite"
NOSQLITE_ENCODER = "lecore_text"
NOSQLITE_INDEX = "embedding_neural"
NOSQLITE_DIMENSIONS = 384
MEMORY_KEY_ENV = "LECORE_X402_MEMORY_KEYS"
MEMORY_MIGRATION_ENV = "LECORE_X402_ALLOW_PLAINTEXT_MIGRATION"
MEMORY_CIPHER = "AES-256-GCM"
MEMORY_COMPRESSION = "zlib"
MEMORY_KDF = "HKDF-SHA256"


LOG = logging.getLogger(__name__)

SERVICE_NAME = "leCore Agent Memory & Routing API"
HERO_TITLE = "Agent memory and routing, paid per call."
X402_BUYER_GUIDE_URL = "https://docs.x402.org/getting-started/quickstart-for-buyers"
API_DESCRIPTION = """Hosted, encrypted tenant-scoped agent memory, capability
routing, and readiness data over HTTPS, with x402 payment on each protected request.

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
`POST /v1/memory` also requires an `Idempotency-Key`, refuses shared-public writes,
and stores the private record through compressed authenticated encryption.
`GET /v1/memory` provides bounded cursor pagination or exact-id retrieval, and
`PATCH /v1/memory` atomically replaces selected fields without rewriting a no-op, while
`DELETE /v1/memory` removes one private record idempotently without resurrection.
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
    PaidRoute("POST", "/v1/memory", "Store one entry in encrypted tenant-scoped agent memory"),
    PaidRoute("GET", "/v1/memory", "List or retrieve encrypted tenant-scoped agent memory"),
    PaidRoute("PATCH", "/v1/memory", "Update one entry in encrypted tenant-scoped agent memory"),
    PaidRoute("DELETE", "/v1/memory", "Delete one entry from encrypted tenant-scoped agent memory"),
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
    idempotency_conflict: bool = False,
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
    if idempotency_conflict:
        responses[409] = _error_response(
            "The idempotency key was already used for different memory content.",
            "Idempotency-Key was already used for a different memory write",
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
    encrypted_storage: bool,
    plaintext_migration_enabled: bool,
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
                "required": ["backend", "nosqlite_shadow", "nosqlite_configured", "durable_transactions", "storage"],
                "properties": {
                    "backend": {"type": "string", "enum": ["core", "nosqlite"]},
                    "nosqlite_shadow": {"type": "boolean"},
                    "nosqlite_configured": {"type": "boolean"},
                    "durable_transactions": {"type": "boolean"},
                    "storage": {
                        "type": "object",
                        "required": ["durable", "encrypted", "cipher", "compression", "plaintext_migration_enabled"],
                        "properties": {
                            "durable": {"type": "boolean"},
                            "encrypted": {"type": "boolean"},
                            "cipher": {"anyOf": [{"type": "string", "const": MEMORY_CIPHER}, {"type": "null"}]},
                            "compression": {"anyOf": [{"type": "string", "const": MEMORY_COMPRESSION}, {"type": "null"}]},
                            "plaintext_migration_enabled": {"type": "boolean"},
                        },
                        "additionalProperties": False,
                    },
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
            "storage": {
                "durable": durable_transactions,
                "encrypted": encrypted_storage,
                "cipher": MEMORY_CIPHER if encrypted_storage else None,
                "compression": MEMORY_COMPRESSION if encrypted_storage else None,
                "plaintext_migration_enabled": plaintext_migration_enabled,
            },
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
    encrypted_storage: bool,
    plaintext_migration_enabled: bool,
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
                "required": ["backend", "nosqlite_shadow", "nosqlite_configured", "durable_transactions", "storage"],
                "properties": {
                    "backend": {"type": "string", "enum": ["core", "nosqlite"]},
                    "nosqlite_shadow": {"type": "boolean"},
                    "nosqlite_configured": {"type": "boolean"},
                    "durable_transactions": {"type": "boolean"},
                    "storage": {
                        "type": "object",
                        "required": ["durable", "encrypted", "cipher", "compression", "plaintext_migration_enabled"],
                        "properties": {
                            "durable": {"type": "boolean"},
                            "encrypted": {"type": "boolean"},
                            "cipher": {"anyOf": [{"type": "string", "const": MEMORY_CIPHER}, {"type": "null"}]},
                            "compression": {"anyOf": [{"type": "string", "const": MEMORY_COMPRESSION}, {"type": "null"}]},
                            "plaintext_migration_enabled": {"type": "boolean"},
                        },
                        "additionalProperties": False,
                    },
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
            "storage": {
                "durable": durable_transactions,
                "encrypted": encrypted_storage,
                "cipher": MEMORY_CIPHER if encrypted_storage else None,
                "compression": MEMORY_COMPRESSION if encrypted_storage else None,
                "plaintext_migration_enabled": plaintext_migration_enabled,
            },
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


def memory_write_success_openapi() -> Dict[str, Any]:
    """Document the successful private-tenant memory write response."""
    schema = {
        "type": "object",
        "required": ["ok", "tenant", "memory", "transaction"],
        "properties": {
            "ok": {"type": "boolean", "const": True},
            "tenant": {"type": "string"},
            "memory": {
                "type": "object",
                "required": ["id", "text", "label", "metadata"],
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "label": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "metadata": {"type": "object", "additionalProperties": True},
                },
                "additionalProperties": False,
            },
            "transaction": {
                "type": "object",
                "required": ["id", "state", "idempotent"],
                "properties": {
                    "id": {"type": "string"},
                    "state": {"type": "string", "enum": ["complete", "core_committed"]},
                    "idempotent": {"type": "boolean", "const": True},
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    }
    example = {
        "ok": True,
        "tenant": "acme",
        "memory": {
            "id": "tx_8f1e6aaf6dcb44f0a8a6434f135c6338",
            "text": "The customer prefers concise release notes.",
            "label": "preference",
            "metadata": {"source": "agent-session"},
        },
        "transaction": {
            "id": "8f1e6aaf6dcb44f0a8a6434f135c6338f31821a279d4865476d08db1b2cecf0f",
            "state": "complete",
            "idempotent": True,
        },
    }
    return _json_success_response("Memory was durably stored once.", schema, example)


def memory_list_success_openapi() -> Dict[str, Any]:
    """Document private-tenant memory listing and direct lookup."""
    item = {
        "type": "object",
        "required": ["id", "text", "label", "metadata"],
        "properties": {
            "id": {"type": "string"},
            "text": {"type": "string"},
            "label": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "metadata": {"type": "object", "additionalProperties": True},
        },
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "required": ["ok", "tenant", "items", "next_cursor"],
        "properties": {
            "ok": {"type": "boolean", "const": True},
            "tenant": {"type": "string"},
            "items": {"type": "array", "items": item, "maxItems": 100},
            "next_cursor": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "additionalProperties": False,
    }
    example = {
        "ok": True,
        "tenant": "acme",
        "items": [{
            "id": "tx_8f1e6aaf6dcb44f0a8a6434f135c6338",
            "text": "The customer prefers concise release notes.",
            "label": "preference",
            "metadata": {"source": "agent-session"},
        }],
        "next_cursor": None,
    }
    return _json_success_response("Private-tenant memory page returned.", schema, example)


def memory_delete_success_openapi() -> Dict[str, Any]:
    """Document idempotent private-tenant memory deletion."""
    schema = {
        "type": "object",
        "required": ["ok", "tenant", "memory_id", "deleted"],
        "properties": {
            "ok": {"type": "boolean", "const": True},
            "tenant": {"type": "string"},
            "memory_id": {"type": "string"},
            "deleted": {"type": "boolean"},
        },
        "additionalProperties": False,
    }
    example = {
        "ok": True,
        "tenant": "acme",
        "memory_id": "tx_8f1e6aaf6dcb44f0a8a6434f135c6338",
        "deleted": True,
    }
    return _json_success_response("Memory deletion completed idempotently.", schema, example)


def memory_update_success_openapi() -> Dict[str, Any]:
    """Document a successful private-tenant memory update."""
    response = memory_write_success_openapi()
    schema = response["content"]["application/json"]["schema"]
    schema["required"].remove("transaction")
    schema["properties"].pop("transaction")
    response["description"] = "Memory fields were updated atomically."
    response["content"]["application/json"]["example"] = {
        "ok": True,
        "tenant": "acme",
        "memory": {
            "id": "tx_8f1e6aaf6dcb44f0a8a6434f135c6338",
            "text": "The customer prefers concise release notes and changelogs.",
            "label": "preference",
            "metadata": {"source": "agent-session", "confirmed": True},
        },
    }
    return response


def memory_update_request_openapi() -> Dict[str, Any]:
    """Document a partial update that requires at least one mutable field."""
    request = paid_request_openapi(
        required=[],
        properties={
            "text": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_MEMORY_CHARS,
                "pattern": r"\S",
                "description": "Replacement memory content; omit to preserve it.",
            },
            "label": {
                "anyOf": [
                    {"type": "string", "maxLength": MAX_MEMORY_LABEL_CHARS},
                    {"type": "null"},
                ],
                "description": "Replacement category; null clears it.",
            },
            "metadata": {
                "type": "object",
                "additionalProperties": True,
                "description": "Replacement JSON metadata; an empty object clears it.",
            },
        },
        example={
            "text": "The customer prefers concise release notes and changelogs.",
            "metadata": {"source": "agent-session", "confirmed": True},
        },
        example_summary="Update selected fields of one private memory",
    )
    schema = request["requestBody"]["content"]["application/json"]["schema"]
    schema.pop("required")
    schema["anyOf"] = [
        {"required": ["text"]},
        {"required": ["label"]},
        {"required": ["metadata"]},
    ]
    schema["additionalProperties"] = False
    return request


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
<div class="copy"><p class="status"><span>$price_per_request</span><span>$environment_label</span><span>$network_label</span></p><h1 id="hero-title">$hero_title</h1><p class="lede">A hosted HTTPS API for storing and recalling encrypted private-tenant memory, routing tasks to leCore capabilities, and reading service readiness. $payment_notice</p><div class="actions"><a class="button primary" href="#quickstart">Make the first request</a><a class="button secondary" href="/docs">Explore API docs</a><a class="button secondary" href="/pricing">View pricing</a></div></div>
<aside class="terminal" aria-label="Unsigned x402 request example"><div class="terminal-top"><span></span><span></span><span></span></div><pre>curl -i $public_url/v1/dashboard

HTTP/2 402 Payment Required
Payment-Required: &lt;base64 challenge&gt;

# Sign, then retry with:
Payment-Signature: &lt;base64 payment&gt;</pre></aside>
</section>
<section class="strip" aria-label="Deployment details"><div><strong>Endpoint</strong><span><a href="$public_url">$public_url</a></span></div><div><strong>Stage</strong><span>$environment_label</span></div><div><strong>Protocol</strong><span>x402 v2</span></div></section>
<section id="quickstart" class="section quickstart" tabindex="-1"><div class="quick-grid"><div><p class="eyebrow">Four-step quickstart</p><h2>Inspect the terms before signing anything.</h2><ol class="flow"><li><div><strong>Read the free manifest</strong><p><a href="/pricing">GET /pricing</a> returns the exact route, network, asset, receiver, and price.</p></div></li><li><div><strong>Make an unsigned request</strong><p>The protected route returns <code>402</code> with a base64 <code>Payment-Required</code> challenge.</p></div></li><li><div><strong>Sign with an x402 v2 client</strong><p>Use the <a href="$buyer_guide_url">x402 buyer guide</a> to configure a testnet wallet and payment client.</p></div></li><li><div><strong>Retry and verify settlement</strong><p>Send <code>Payment-Signature</code>; a successful response includes <code>Payment-Response</code>.</p></div></li></ol></div><div class="command-panel"><div class="command-head"><span>First request: no wallet required</span><a href="/docs#/Paid%20API/getDashboard">Open route docs</a></div><pre><code>curl -i $public_url/v1/dashboard</code></pre><p class="response-note"><strong>Expected:</strong> HTTP 402 plus <code>Payment-Required</code>. This safely exposes the payment contract without moving testnet funds.</p><div class="command-head"><span>Inspect exact preview terms</span><a href="/pricing">Open live JSON</a></div><pre><code>curl -sS $public_url/pricing</code></pre></div></div></section>
<section id="routes" class="section"><div class="heading"><p class="eyebrow">Paid API surface</p><h2>A complete private-memory lifecycle, with no subscription.</h2></div><div class="routes"><article class="card"><p class="method">POST · GET · PATCH · DELETE</p><h3>Manage memory</h3><code>/v1/memory</code><p>Store, page, retrieve, update, and idempotently delete encrypted private-tenant memories.</p></article><article class="card"><p class="method">POST</p><h3>Recall</h3><code>/v1/recall</code><p>Query the seeded public preview memory or your authenticated private tenant.</p></article><article class="card"><p class="method">POST</p><h3>Route</h3><code>/v1/route</code><p>Send a plain-language task and receive an explicit act, choose, or unknown decision with evidence.</p></article><article class="card"><p class="method">GET</p><h3>Dashboard</h3><code>/v1/dashboard</code><p>Read memory, capability-routing, and deterministic-engine readiness for one tenant.</p></article></div></section>
<section id="proof" class="section proof"><div class="proof-copy"><p class="eyebrow">Storage boundaries</p><h2>Encrypted before durable memory reaches disk.</h2><p>Private-tenant memory and its retry journal are compressed, encrypted with per-record AES-256-GCM keys derived from a versioned service key, and authenticated against their tenant and file identity. The shared public dataset stays read-only. Private tenants still require operator-issued access credentials.</p></div><div class="proof-panel"><dl><div><dt>Per request</dt><dd>$price_per_request</dd></div><div><dt>Per 1,000</dt><dd>$price_per_thousand</dd></div><div><dt>Network</dt><dd>$network_name</dd></div><div><dt>API version</dt><dd>$api_version</dd></div></dl></div></section>
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


def normalize_memory_id(value: Any) -> str:
    """Validate a memory id before lookup or deletion."""
    if not isinstance(value, str) or not _MEMORY_ID_RE.fullmatch(value):
        raise ValueError("memory_id must be 1-128 letters, numbers, '.', '_', ':', or '-'")
    return value


@contextmanager
def _process_file_lock(path: Path) -> Any:
    """Hold an exclusive process lock for one persisted tenant state file."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.is_symlink():
        raise MemoryStateError("durable-state lock must not be a symbolic link")
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


class MemoryStateError(RuntimeError):
    """Durable memory could not be authenticated, decoded, or migrated safely."""


@dataclass(frozen=True)
class MemoryKeyring:
    """A small versioned set of 256-bit application data-encryption keys."""

    active: str
    keys: Dict[str, bytes]

    @classmethod
    def from_json(cls, value: str) -> "MemoryKeyring":
        """Parse the Secrets Manager value used by ``LECORE_X402_MEMORY_KEYS``."""
        try:
            document = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("%s must be a JSON object" % MEMORY_KEY_ENV) from exc
        if not isinstance(document, dict) or set(document) != {"active", "keys"}:
            raise ValueError("%s must contain exactly 'active' and 'keys'" % MEMORY_KEY_ENV)
        active = document.get("active")
        encoded_keys = document.get("keys")
        if not isinstance(active, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", active):
            raise ValueError("memory active key id must be 1-64 safe characters")
        if not isinstance(encoded_keys, dict) or not 1 <= len(encoded_keys) <= 8:
            raise ValueError("memory keyring must contain between 1 and 8 keys")
        keys: Dict[str, bytes] = {}
        for key_id, encoded in encoded_keys.items():
            if not isinstance(key_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", key_id):
                raise ValueError("memory key ids must be 1-64 safe characters")
            if not isinstance(encoded, str) or not encoded:
                raise ValueError("memory key %s must be base64" % key_id)
            try:
                padded = encoded + "=" * (-len(encoded) % 4)
                key = base64.b64decode(padded, altchars=b"-_", validate=True)
            except (ValueError, TypeError) as exc:
                raise ValueError("memory key %s must be valid base64" % key_id) from exc
            if len(key) != 32:
                raise ValueError("memory key %s must decode to exactly 32 bytes" % key_id)
            keys[key_id] = key
        if active not in keys:
            raise ValueError("memory active key id is not present in the keyring")
        if len(set(keys.values())) != len(keys):
            raise ValueError("memory key ids must not contain duplicate key material")
        return cls(active=active, keys=keys)


class MemoryStateCodec:
    """Compress and authenticate durable JSON records before they touch disk.

    The versioned envelope deliberately leaves only format metadata and the key
    id visible. Tenant/file identity is authenticated as associated data, so an
    encrypted record cannot be copied into another tenant or journal location.
    """

    _MAGIC = b"LECMEM01"
    _VERSION = 1
    _MAX_PLAINTEXT_BYTES = 256 * 1024 * 1024
    _MAX_HEADER_BYTES = 4096

    def __init__(self, keyring: MemoryKeyring, allow_plaintext_migration: bool = False):
        self.keyring = keyring
        self.allow_plaintext_migration = bool(allow_plaintext_migration)

    def read_json(self, path: Path, context: str) -> Dict[str, Any]:
        """Read one record, migrating plaintext or an old key while locked."""
        try:
            if path.is_symlink() or not path.is_file():
                raise MemoryStateError("durable state %s must be a regular file" % path.name)
            if path.stat().st_size > self._MAX_PLAINTEXT_BYTES + self._MAX_HEADER_BYTES + 64:
                raise MemoryStateError("durable state %s exceeds the 256 MiB safety limit" % path.name)
            payload = path.read_bytes()
        except OSError as exc:
            raise MemoryStateError("could not read durable memory state %s" % path.name) from exc
        if payload.startswith(self._MAGIC):
            value, key_id = self._decrypt_json(payload, context, path.name)
            if key_id != self.keyring.active:
                self.write_json(path, value, context)
            return value
        if not self.allow_plaintext_migration:
            raise MemoryStateError(
                "plaintext durable state %s is refused; enable one-time migration explicitly" % path.name
            )
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MemoryStateError("invalid plaintext durable state %s" % path.name) from exc
        if not isinstance(value, dict):
            raise MemoryStateError("durable state %s is not a JSON object" % path.name)
        self.write_json(path, value, context)
        return value

    def write_json(self, path: Path, value: Dict[str, Any], context: str) -> None:
        """Serialize, compress, encrypt, and atomically replace one record."""
        if path.is_symlink():
            raise MemoryStateError("durable state %s must not be a symbolic link" % path.name)
        if not isinstance(value, dict):
            raise MemoryStateError("durable memory state must be a JSON object")
        try:
            plaintext = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise MemoryStateError("durable memory state is not JSON serializable") from exc
        if len(plaintext) > self._MAX_PLAINTEXT_BYTES:
            raise MemoryStateError("durable memory state exceeds the 256 MiB safety limit")
        compressed = zlib.compress(plaintext, level=6)
        nonce = os.urandom(12)
        header = {
            "algorithm": MEMORY_CIPHER,
            "compression": MEMORY_COMPRESSION,
            "kdf": MEMORY_KDF,
            "key_id": self.keyring.active,
            "nonce": base64.urlsafe_b64encode(nonce).decode("ascii").rstrip("="),
            "plaintext_bytes": len(plaintext),
            "version": self._VERSION,
        }
        header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii")
        associated_data = self._associated_data(header_bytes, context)
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:  # pragma: no cover - covered by the x402 install
            raise RuntimeError("memory encryption requires cryptography>=46,<47") from exc
        ciphertext = AESGCM(self._data_key(self.keyring.keys[self.keyring.active], context)).encrypt(
            nonce,
            compressed,
            associated_data,
        )
        envelope = self._MAGIC + struct.pack(">I", len(header_bytes)) + header_bytes + ciphertext
        _atomic_write_bytes(path, envelope)

    def _decrypt_json(self, payload: bytes, context: str, name: str) -> Tuple[Dict[str, Any], str]:
        try:
            if len(payload) < len(self._MAGIC) + 4:
                raise ValueError("truncated header")
            offset = len(self._MAGIC)
            header_length = struct.unpack(">I", payload[offset:offset + 4])[0]
            if not 1 <= header_length <= self._MAX_HEADER_BYTES:
                raise ValueError("invalid header length")
            header_start = offset + 4
            header_end = header_start + header_length
            header_bytes = payload[header_start:header_end]
            ciphertext = payload[header_end:]
            if len(ciphertext) < 16:
                raise ValueError("truncated ciphertext")
            header = json.loads(header_bytes.decode("ascii"))
            if not isinstance(header, dict) or set(header) != {
                "algorithm", "compression", "kdf", "key_id", "nonce", "plaintext_bytes", "version"
            }:
                raise ValueError("invalid header")
            if (
                header["version"] != self._VERSION
                or header["algorithm"] != MEMORY_CIPHER
                or header["compression"] != MEMORY_COMPRESSION
                or header["kdf"] != MEMORY_KDF
            ):
                raise ValueError("unsupported envelope")
            key_id = header["key_id"]
            if key_id not in self.keyring.keys:
                raise MemoryStateError("durable state %s needs unavailable memory key %s" % (name, key_id))
            encoded_nonce = header["nonce"]
            if not isinstance(encoded_nonce, str):
                raise ValueError("invalid nonce")
            nonce = base64.b64decode(
                encoded_nonce + "=" * (-len(encoded_nonce) % 4),
                altchars=b"-_",
                validate=True,
            )
            if len(nonce) != 12:
                raise ValueError("invalid nonce")
            expected_size = header["plaintext_bytes"]
            if not isinstance(expected_size, int) or not 0 <= expected_size <= self._MAX_PLAINTEXT_BYTES:
                raise ValueError("invalid plaintext size")
        except MemoryStateError:
            raise
        except (ValueError, TypeError, KeyError, UnicodeDecodeError, json.JSONDecodeError, struct.error) as exc:
            raise MemoryStateError("invalid encrypted durable state %s" % name) from exc
        try:
            from cryptography.exceptions import InvalidTag
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            compressed = AESGCM(self._data_key(self.keyring.keys[key_id], context)).decrypt(
                nonce,
                ciphertext,
                self._associated_data(header_bytes, context),
            )
        except InvalidTag as exc:
            raise MemoryStateError("durable state authentication failed for %s" % name) from exc
        plaintext = self._decompress(compressed, name)
        if len(plaintext) != expected_size:
            raise MemoryStateError("durable state size check failed for %s" % name)
        try:
            value = json.loads(plaintext.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MemoryStateError("invalid encrypted JSON in %s" % name) from exc
        if not isinstance(value, dict):
            raise MemoryStateError("durable state %s is not a JSON object" % name)
        return value, key_id

    @classmethod
    def _decompress(cls, compressed: bytes, name: str) -> bytes:
        try:
            inflater = zlib.decompressobj()
            plaintext = inflater.decompress(compressed, cls._MAX_PLAINTEXT_BYTES + 1)
            if len(plaintext) > cls._MAX_PLAINTEXT_BYTES or inflater.unconsumed_tail:
                raise MemoryStateError("decompressed durable state %s exceeds the safety limit" % name)
            plaintext += inflater.flush(cls._MAX_PLAINTEXT_BYTES + 1 - len(plaintext))
            if len(plaintext) > cls._MAX_PLAINTEXT_BYTES or not inflater.eof or inflater.unused_data:
                raise MemoryStateError("invalid compressed durable state %s" % name)
            return plaintext
        except zlib.error as exc:
            raise MemoryStateError("invalid compressed durable state %s" % name) from exc

    @classmethod
    def _associated_data(cls, header: bytes, context: str) -> bytes:
        if not isinstance(context, str) or not context or len(context.encode("utf-8")) > 512:
            raise MemoryStateError("invalid durable-state encryption context")
        return cls._MAGIC + header + b"\0" + context.encode("utf-8")

    @staticmethod
    def _data_key(master_key: bytes, context: str) -> bytes:
        """Derive a distinct AES key for each tenant/file context."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF

        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"lecore-x402-memory-v1\0" + context.encode("utf-8"),
        ).derive(master_key)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Durably replace one small state record with owner-only permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%s.tmp" % (path.name, os.urandom(8).hex()))
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        if hasattr(os, "O_DIRECTORY"):
            directory = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    """Durably replace a small JSON control record without exposing a partial file."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    _atomic_write_bytes(path, payload)


class TenantCoreStore:
    """Thread-safe LocalAgentCore registry with optional per-tenant persistence."""

    def __init__(
        self,
        default_core: LocalAgentCore,
        state_dir: Optional[Any] = None,
        codec: Optional[MemoryStateCodec] = None,
    ):
        self._default_dim = default_core.dim
        self._default_seed = default_core.seed
        self._default_route_threshold = default_core.route_threshold
        self._cores: Dict[str, LocalAgentCore] = {DEFAULT_TENANT_ID: default_core}
        self._versions: Dict[str, Tuple[int, int, int]] = {}
        self._tenant_locks: Dict[str, threading.RLock] = {DEFAULT_TENANT_ID: threading.RLock()}
        self._registry_lock = threading.RLock()
        self._state_dir = Path(state_dir) if state_dir else None
        self._codec = codec
        if self._state_dir is not None:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            if self._codec is not None:
                self._migrate_and_rotate_all()
            public_path = self._path_for(DEFAULT_TENANT_ID)
            if public_path is not None and public_path.exists():
                with _process_file_lock(public_path):
                    self._cores[DEFAULT_TENANT_ID] = self._load_core(public_path, DEFAULT_TENANT_ID)
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
        return self.mutate(tenant_id, lambda core: (fn(core), True))

    def mutate(self, tenant_id: str, fn: Any) -> Any:
        """Persist a callback result only when it reports that state changed.

        When the cached file version is current, mutate it in place and keep a
        rollback snapshot. This avoids decrypting and rebuilding the same
        tenant index before every single-process write while remaining safe if
        another task changed the file or persistence fails.
        """
        normalized = normalize_tenant_id(tenant_id)
        with self._lock_for(normalized):
            path = self._path_for(normalized)
            if path is None:
                core = self._get_cached(normalized)
                result, _changed = fn(core)
                return result
            with _process_file_lock(path):
                borrowed_cached = False
                original_state: Optional[Dict[str, Any]] = None
                current_version = self._version(path) if path.exists() else None
                with self._registry_lock:
                    cached = self._cores.get(normalized)
                    cached_version = self._versions.get(normalized)
                if cached is not None and current_version is not None and cached_version == current_version:
                    core = cached
                    borrowed_cached = True
                    original_state = core.to_state()
                elif path.exists():
                    core = self._load_core(path, normalized)
                else:
                    core = LocalAgentCore.from_state(self._get_cached(normalized).to_state())
                try:
                    result, changed = fn(core)
                    if changed:
                        self._save_core(path, normalized, core)
                except Exception:
                    if borrowed_cached and original_state is not None:
                        with self._registry_lock:
                            self._cores[normalized] = LocalAgentCore.from_state(original_state)
                    raise
                with self._registry_lock:
                    self._cores[normalized] = core
                    if path.exists():
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
                    core = self._load_core(path, tenant_id)
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

    def _load_core(self, path: Path, tenant_id: str) -> LocalAgentCore:
        if self._codec is None:
            return LocalAgentCore.load(path)
        state = self._codec.read_json(path, "core:%s" % normalize_tenant_id(tenant_id))
        return LocalAgentCore.from_state(state)

    def _save_core(self, path: Path, tenant_id: str, core: LocalAgentCore) -> None:
        if self._codec is None:
            core.save(path)
            return
        self._codec.write_json(path, core.to_state(), "core:%s" % normalize_tenant_id(tenant_id))

    def _migrate_and_rotate_all(self) -> None:
        """Rewrite every existing tenant file under the active authenticated key."""
        if self._state_dir is None or self._codec is None:
            return
        for path in sorted(self._state_dir.glob("*.json")):
            if path.is_symlink() or not path.is_file():
                raise MemoryStateError("durable tenant state must be a regular file: %s" % path.name)
            tenant_id = normalize_tenant_id(path.stem)
            if tenant_id != path.stem:
                raise MemoryStateError("durable tenant filename is not canonical: %s" % path.name)
            with _process_file_lock(path):
                self._codec.read_json(path, "core:%s" % tenant_id)


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

    def replace(self, tenant_id: str, memory: Dict[str, Any]) -> None:
        """Idempotently replace one projected memory, including its embedding."""
        normalized = normalize_tenant_id(tenant_id)
        with self._lock:
            collection = self._ensure_collection(normalized)
            try:
                self._delete_memory(collection, memory["id"])
                self._insert_memory(collection, normalized, memory)
            except NoSQLiteError:
                self._synced_collections.discard((self._process.generation, collection))
                raise

    def delete(self, tenant_id: str, memory_id: str) -> None:
        """Idempotently delete one projected memory."""
        normalized = normalize_tenant_id(tenant_id)
        with self._lock:
            collection = self._ensure_collection(normalized)
            try:
                self._delete_memory(collection, memory_id)
            except NoSQLiteError:
                self._synced_collections.discard((self._process.generation, collection))
                raise

    def sync(self, tenant_id: str, memories: Iterable[Dict[str, Any]]) -> None:
        """Reconcile a tenant projection from its authoritative encrypted snapshot."""
        normalized = normalize_tenant_id(tenant_id)
        with self._lock:
            collection = self._ensure_collection(normalized)
            key = (self._process.generation, collection)
            if key in self._synced_collections:
                return
            try:
                self._process.command({"delete": collection})
                for memory in memories:
                    self._insert_memory(collection, normalized, memory)
            except NoSQLiteError:
                self._synced_collections.discard(key)
                raise
            else:
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

    def _delete_memory(self, collection: str, memory_id: Any) -> None:
        self._process.command({
            "delete": collection,
            "filter": {"_id": str(memory_id)},
        })

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
    _DELETED = "deleted"

    def __init__(
        self,
        core_store: TenantCoreStore,
        state_dir: Any,
        codec: Optional[MemoryStateCodec] = None,
    ):
        self._core_store = core_store
        self._root = Path(state_dir) / ".x402-memory-transactions"
        self._codec = codec
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
                    if record.get("state") in {self._COMPLETE, self._DELETED}:
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
        if record["state"] == self._DELETED:
            return self._result(record, memory)
        stored = self._core_store.write(
            record["tenant"],
            lambda core: self._ensure_core_memory(core, memory),
        )
        if record["state"] == self._PLANNED:
            record["state"] = self._CORE_COMMITTED
            self._write(path, record)

        if record["requires_mirror"]:
            if mirror is None:
                return self._result(record, stored)
            try:
                mirror.remember(record["tenant"], stored)
            except NoSQLiteError as exc:
                raise MemoryMirrorPending(record["tenant"], record["transaction_id"], exc) from exc

        if record["state"] != self._COMPLETE:
            record["state"] = self._COMPLETE
            self._write(path, record)
        return self._result(record, stored)

    def mark_deleted(self, tenant_id: str, memory_id: str) -> bool:
        """Tombstone the originating idempotency record before deleting memory."""
        tenant = normalize_tenant_id(tenant_id)
        wanted = normalize_memory_id(memory_id)
        if not wanted.startswith("tx_"):
            return False
        prefix = wanted[3:]
        directory = self._root / self._hash(tenant)[:24]
        for path in sorted(directory.glob(prefix + "*.json")):
            with _process_file_lock(path):
                record = self._load(path)
                self._validate_record(record, path)
                if record["tenant"] != tenant or record["memory"].get("id") != wanted:
                    continue
                if record["state"] != self._DELETED:
                    record["state"] = self._DELETED
                    self._write(path, record)
                return True
        return False

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
        self._write(path, record)
        return record

    def _load(self, path: Path) -> Dict[str, Any]:
        if self._codec is not None:
            return self._codec.read_json(path, self._encryption_context(path))
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MemoryTransactionError("invalid transaction journal %s" % path.name) from exc
        if not isinstance(value, dict):
            raise MemoryTransactionError("transaction journal %s is not an object" % path.name)
        return value

    def _write(self, path: Path, record: Dict[str, Any]) -> None:
        if self._codec is None:
            _atomic_write_json(path, record)
            return
        self._codec.write_json(path, record, self._encryption_context(path))

    def _encryption_context(self, path: Path) -> str:
        try:
            relative = path.relative_to(self._root)
        except ValueError as exc:  # pragma: no cover - paths are constructed internally
            raise MemoryStateError("transaction journal escaped its state directory") from exc
        if len(relative.parts) != 2:
            raise MemoryStateError("invalid transaction journal path")
        return "journal:%s/%s" % (relative.parts[0], relative.parts[1])

    def _validate_record(self, record: Dict[str, Any], path: Path) -> None:
        required = {"version", "transaction_id", "tenant", "request_fingerprint", "requires_mirror", "state", "memory"}
        if not required.issubset(record) or record.get("version") != self._VERSION:
            raise MemoryTransactionError("unsupported transaction journal %s" % path.name)
        if record["state"] not in {self._PLANNED, self._CORE_COMMITTED, self._COMPLETE, self._DELETED}:
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


def memory_state_codec(
    value: Optional[Any],
    *,
    allow_plaintext_migration: bool = False,
) -> Optional[MemoryStateCodec]:
    """Resolve optional versioned encryption material without a silent fallback."""
    if value is None or value == "":
        if allow_plaintext_migration:
            raise ValueError("plaintext migration requires %s" % MEMORY_KEY_ENV)
        return None
    if isinstance(value, MemoryKeyring):
        keyring = value
    elif isinstance(value, str):
        keyring = MemoryKeyring.from_json(value)
    else:
        raise ValueError("memory keys must be a MemoryKeyring or JSON string")
    return MemoryStateCodec(keyring, allow_plaintext_migration=allow_plaintext_migration)


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
    memory_keys: Optional[Any] = None,
    allow_plaintext_migration: Optional[bool] = None,
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
        from fastapi import FastAPI, Header, HTTPException, Query
        from fastapi.responses import HTMLResponse
    except ImportError as exc:
        raise RuntimeError(optional_dependency_help()) from exc

    config = config or (X402Config.from_env(require_pay_to=paid) if paid else X402Config.from_env(require_pay_to=False))
    public = urlsplit(config.public_url)
    if paid and public.scheme != "https" and public.hostname not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("paid mode public_url must use https outside localhost")

    allow_plaintext_migration = (
        bool(allow_plaintext_migration)
        if allow_plaintext_migration is not None
        else env_flag(os.environ.get(MEMORY_MIGRATION_ENV))
    )
    codec = memory_state_codec(
        memory_keys if memory_keys is not None else os.environ.get(MEMORY_KEY_ENV),
        allow_plaintext_migration=allow_plaintext_migration,
    )
    durable_write_published = any(route.path == "/v1/memory" for route in config.routes)
    if paid and durable_write_published and not tenant_state_dir:
        raise ValueError("paid /v1/memory requires LECORE_X402_TENANT_STATE_DIR")
    if paid and tenant_state_dir and codec is None:
        raise ValueError("paid durable memory requires %s" % MEMORY_KEY_ENV)

    core = core or demo()
    store = TenantCoreStore(core, state_dir=tenant_state_dir, codec=codec)
    memory_backend = normalize_memory_backend(
        memory_backend if memory_backend is not None else os.environ.get("LECORE_X402_MEMORY_BACKEND", MEMORY_BACKEND_CORE)
    )
    nosqlite_shadow = (
        bool(nosqlite_shadow)
        if nosqlite_shadow is not None
        else env_flag(os.environ.get("LECORE_X402_NOSQLITE_SHADOW"))
    )
    if codec is not None and (memory_backend == MEMORY_BACKEND_NOSQLITE or nosqlite_shadow):
        raise ValueError("encrypted memory does not permit the plaintext NoSQLite backend or shadow")
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
    memory_transactions = TenantMemoryTransactions(store, tenant_state_dir, codec=codec) if tenant_state_dir else None

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
    app.state.memory_state_codec = codec
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
            "storage": {
                "durable": bool(tenant_state_dir),
                "encrypted": codec is not None,
                "cipher": MEMORY_CIPHER if codec is not None else None,
                "compression": MEMORY_COMPRESSION if codec is not None else None,
                "plaintext_migration_enabled": bool(codec and codec.allow_plaintext_migration),
            },
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
                encrypted_storage=codec is not None,
                plaintext_migration_enabled=bool(codec and codec.allow_plaintext_migration),
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
                encrypted_storage=codec is not None,
                plaintext_migration_enabled=bool(codec and codec.allow_plaintext_migration),
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

    def memory_fields(payload: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
        text = validated(_required_text, payload, "text", MAX_MEMORY_CHARS)
        label = payload.get("label")
        metadata = payload.get("metadata")
        if label is not None:
            if not isinstance(label, str):
                raise HTTPException(status_code=400, detail="label must be a string")
            if len(label) > MAX_MEMORY_LABEL_CHARS:
                raise HTTPException(
                    status_code=400,
                    detail="label must be at most %d characters" % MAX_MEMORY_LABEL_CHARS,
                )
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise HTTPException(status_code=400, detail="metadata must be an object")
            encoded_metadata = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
            if len(encoded_metadata) > MAX_MEMORY_METADATA_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail="metadata must be at most %d encoded bytes" % MAX_MEMORY_METADATA_BYTES,
                )
        return text, label, metadata

    def memory_update_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
        allowed = {"text", "label", "metadata"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise HTTPException(status_code=400, detail="unknown memory update field: %s" % unknown[0])
        updates = {key: payload[key] for key in allowed if key in payload}
        if not updates:
            raise HTTPException(status_code=400, detail="at least one of text, label, or metadata is required")
        if "text" in updates:
            updates["text"] = validated(_required_text, updates, "text", MAX_MEMORY_CHARS)
        if "label" in updates:
            label = updates["label"]
            if label is not None and not isinstance(label, str):
                raise HTTPException(status_code=400, detail="label must be a string or null")
            if isinstance(label, str) and len(label) > MAX_MEMORY_LABEL_CHARS:
                raise HTTPException(
                    status_code=400,
                    detail="label must be at most %d characters" % MAX_MEMORY_LABEL_CHARS,
                )
        if "metadata" in updates:
            metadata = updates["metadata"]
            if not isinstance(metadata, dict):
                raise HTTPException(status_code=400, detail="metadata must be an object")
            encoded_metadata = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
            if len(encoded_metadata) > MAX_MEMORY_METADATA_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail="metadata must be at most %d encoded bytes" % MAX_MEMORY_METADATA_BYTES,
                )
        return updates

    def commit_memory(
        tenant_id: str,
        text: str,
        label: Optional[str],
        metadata: Optional[Dict[str, Any]],
        key: Optional[str],
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
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
            if committed["transaction"]["state"] == "deleted":
                raise HTTPException(
                    status_code=409,
                    detail="this idempotent memory was deleted; use a new Idempotency-Key",
                )
            return committed["memory"], committed["transaction"]
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
        return memory, transaction

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
        "/v1/memory",
        tags=["Paid API"],
        operation_id="storeMemory",
        summary="Store private agent memory",
        description=(
            "Durably store one entry in an authenticated private tenant. The write is "
            "compressed, encrypted at the application boundary, and made idempotent by "
            "the required Idempotency-Key. Shared public memory is read-only."
        ),
        responses=paid_operation_responses(
            memory_write_success_openapi(),
            invalid_detail="text and Idempotency-Key are required",
            idempotency_conflict=True,
        ),
        openapi_extra=paid_request_openapi(
            required=["text"],
            properties={
                "text": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_MEMORY_CHARS,
                    "pattern": r"\S",
                    "description": "Memory content to store in the selected private tenant.",
                },
                "label": {
                    "type": "string",
                    "maxLength": MAX_MEMORY_LABEL_CHARS,
                    "description": "Optional caller-defined category.",
                },
                "metadata": {
                    "type": "object",
                    "additionalProperties": True,
                    "description": "Optional JSON metadata, limited to 16 KiB when encoded.",
                },
                "tenant": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 64,
                    "description": "Private tenant id; may instead be supplied in X-leCore-Tenant.",
                },
            },
            example={
                "tenant": "acme",
                "text": "The customer prefers concise release notes.",
                "label": "preference",
                "metadata": {"source": "agent-session"},
            },
            example_summary="Store an idempotent private-tenant memory",
        ),
    )
    def store_memory(
        payload: Dict[str, Any],
        idempotency_key: str = Header(
            ...,
            alias=IDEMPOTENCY_HEADER,
            description="Required stable retry key. Reuse it only for the identical memory write.",
        ),
        x_lecore_tenant: Optional[str] = Header(
            default=None,
            alias=TENANT_HEADER,
            description="Private tenant id, if it is not supplied in the JSON body.",
        ),
        x_lecore_tenant_token: Optional[str] = Header(
            default=None,
            alias=TENANT_TOKEN_HEADER,
            description="Required authorization token for the resolved private tenant.",
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
        tenant_id = tenant_from_payload(payload, x_lecore_tenant)
        if tenant_id == DEFAULT_TENANT_ID:
            raise HTTPException(status_code=403, detail="shared public memory is read-only")
        require_tenant_access(tenant_id, x_lecore_tenant_token)
        if memory_transactions is None or codec is None:
            raise HTTPException(status_code=503, detail="encrypted durable memory is not configured")
        key = validated(normalize_idempotency_key, idempotency_key)
        if key is None:  # pragma: no cover - FastAPI marks the header required
            raise HTTPException(status_code=400, detail="Idempotency-Key is required")
        text, label, metadata = memory_fields(payload)
        memory, transaction = commit_memory(tenant_id, text, label, metadata, key)
        return {
            "ok": True,
            "tenant": tenant_id,
            "memory": memory,
            "transaction": transaction,
        }

    @app.get(
        "/v1/memory",
        tags=["Paid API"],
        operation_id="getMemory",
        summary="List or retrieve private agent memory",
        description=(
            "Return a bounded insertion-ordered page for an authenticated private tenant. "
            "Pass memory_id for one exact record, or cursor and limit for pagination."
        ),
        responses={
            **paid_operation_responses(
                memory_list_success_openapi(),
                invalid_detail="memory_id, cursor, or limit is invalid",
            ),
            404: _error_response("The requested memory does not exist in this tenant.", "memory not found"),
        },
    )
    def get_memory(
        memory_id: Optional[str] = Query(
            default=None,
            description="Return this exact memory id instead of a page.",
        ),
        limit: int = Query(
            default=50,
            ge=1,
            le=100,
            description="Maximum memories in a page.",
        ),
        cursor: Optional[str] = Query(
            default=None,
            description="Last memory id from the previous page.",
        ),
        x_lecore_tenant: str = Header(
            ...,
            alias=TENANT_HEADER,
            description="Required private tenant id.",
        ),
        x_lecore_tenant_token: str = Header(
            ...,
            alias=TENANT_TOKEN_HEADER,
            description="Required authorization token for the private tenant.",
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
        tenant_id = tenant_from_header(x_lecore_tenant)
        if tenant_id == DEFAULT_TENANT_ID:
            raise HTTPException(status_code=403, detail="consumer memory access requires a private tenant")
        require_tenant_access(tenant_id, x_lecore_tenant_token)
        if codec is None or memory_transactions is None:
            raise HTTPException(status_code=503, detail="encrypted durable memory is not configured")
        if memory_id is not None:
            if cursor is not None:
                raise HTTPException(status_code=400, detail="memory_id and cursor cannot be combined")
            wanted = validated(normalize_memory_id, memory_id)
            item = store.read(tenant_id, lambda tenant_core: tenant_core.get_memory(wanted))
            if item is None:
                raise HTTPException(status_code=404, detail="memory not found")
            page = {"items": [item], "next_cursor": None}
        else:
            normalized_cursor = validated(normalize_memory_id, cursor) if cursor is not None else None
            page = validated(
                lambda: store.read(
                    tenant_id,
                    lambda tenant_core: tenant_core.list_memories(limit=limit, cursor=normalized_cursor),
                )
            )
        return {"ok": True, "tenant": tenant_id, **page}

    @app.patch(
        "/v1/memory",
        tags=["Paid API"],
        operation_id="updateMemory",
        summary="Update private agent memory",
        description=(
            "Atomically replace selected fields of one authenticated private-tenant memory. "
            "Omitted fields are preserved; label may be null and an empty metadata object clears metadata."
        ),
        responses={
            **paid_operation_responses(
                memory_update_success_openapi(),
                invalid_detail="memory_id or update fields are invalid",
            ),
            404: _error_response("The requested memory does not exist in this tenant.", "memory not found"),
        },
        openapi_extra=memory_update_request_openapi(),
    )
    def update_memory(
        payload: Dict[str, Any],
        memory_id: str = Query(..., description="Memory id to update atomically."),
        x_lecore_tenant: str = Header(
            ...,
            alias=TENANT_HEADER,
            description="Required private tenant id.",
        ),
        x_lecore_tenant_token: str = Header(
            ...,
            alias=TENANT_TOKEN_HEADER,
            description="Required authorization token for the private tenant.",
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
        tenant_id = tenant_from_header(x_lecore_tenant)
        if tenant_id == DEFAULT_TENANT_ID:
            raise HTTPException(status_code=403, detail="consumer memory updates require a private tenant")
        require_tenant_access(tenant_id, x_lecore_tenant_token)
        if codec is None or memory_transactions is None:
            raise HTTPException(status_code=503, detail="encrypted durable memory is not configured")
        wanted = validated(normalize_memory_id, memory_id)
        updates = memory_update_fields(payload)

        def replace(tenant_core: LocalAgentCore) -> Tuple[Dict[str, Any], bool]:
            before = tenant_core.get_memory(wanted)
            if before is None:
                return {"memory": None, "changed": False}, False
            updated = tenant_core.update_memory(wanted, **updates)
            changed = updated != before
            return {"memory": updated, "changed": changed}, changed

        mutation = store.mutate(tenant_id, replace)
        memory = mutation["memory"]
        if memory is None:
            raise HTTPException(status_code=404, detail="memory not found")
        if mutation["changed"] and nosqlite_store is not None:
            try:
                nosqlite_store.replace(tenant_id, memory)
            except NoSQLiteError as exc:
                if memory_backend == MEMORY_BACKEND_NOSQLITE:
                    raise nosqlite_unavailable(exc) from exc
                LOG.warning("NoSQLite shadow memory update failed: %s", exc)
        return {"ok": True, "tenant": tenant_id, "memory": memory}

    @app.delete(
        "/v1/memory",
        tags=["Paid API"],
        operation_id="deleteMemory",
        summary="Delete private agent memory",
        description=(
            "Idempotently delete one memory from an authenticated private tenant. "
            "Deleting an already-absent id returns deleted=false and does not rewrite storage."
        ),
        responses=paid_operation_responses(
            memory_delete_success_openapi(),
            invalid_detail="memory_id is invalid",
        ),
    )
    def delete_memory(
        memory_id: str = Query(..., description="Memory id to delete idempotently."),
        x_lecore_tenant: str = Header(
            ...,
            alias=TENANT_HEADER,
            description="Required private tenant id.",
        ),
        x_lecore_tenant_token: str = Header(
            ...,
            alias=TENANT_TOKEN_HEADER,
            description="Required authorization token for the private tenant.",
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
        tenant_id = tenant_from_header(x_lecore_tenant)
        if tenant_id == DEFAULT_TENANT_ID:
            raise HTTPException(status_code=403, detail="consumer memory deletion requires a private tenant")
        require_tenant_access(tenant_id, x_lecore_tenant_token)
        if codec is None or memory_transactions is None:
            raise HTTPException(status_code=503, detail="encrypted durable memory is not configured")
        wanted = validated(normalize_memory_id, memory_id)
        memory_transactions.mark_deleted(tenant_id, wanted)

        def remove(tenant_core: LocalAgentCore) -> Tuple[Optional[Dict[str, Any]], bool]:
            removed = tenant_core.forget(wanted)
            return removed, removed is not None

        removed = store.mutate(tenant_id, remove)
        if nosqlite_store is not None:
            try:
                nosqlite_store.delete(tenant_id, wanted)
            except NoSQLiteError as exc:
                if memory_backend == MEMORY_BACKEND_NOSQLITE:
                    raise nosqlite_unavailable(exc) from exc
                LOG.warning("NoSQLite shadow memory deletion failed: %s", exc)
        return {
            "ok": True,
            "tenant": tenant_id,
            "memory_id": wanted,
            "deleted": removed is not None,
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
        text, label, metadata = memory_fields(payload)
        key = validated(normalize_idempotency_key, idempotency_key)
        memory, transaction = commit_memory(tenant_id, text, label, metadata, key)
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
