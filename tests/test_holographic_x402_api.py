"""Tests for the optional x402-paid API publisher."""

import json
import base64
from html import escape
import os
from pathlib import Path
import threading

import pytest

from holographic_x402_api import (
    DEFAULT_NETWORK,
    DEFAULT_PRICE,
    DEFAULT_PUBLIC_URL,
    DEFAULT_TENANT_ID,
    HERO_TITLE,
    IDEMPOTENCY_HEADER,
    MEMORY_BACKEND_NOSQLITE,
    MEMORY_CIPHER,
    MEMORY_COMPRESSION,
    MEMORY_KEY_ENV,
    MemoryKeyring,
    MemoryStateCodec,
    MemoryStateError,
    MemoryTransactionConflict,
    MemoryMirrorPending,
    NoSQLiteError,
    NoSQLiteMemoryStore,
    SERVICE_NAME,
    TENANT_HEADER,
    TENANT_TOKEN_HEADER,
    TenantCoreStore,
    TenantMemoryTransactions,
    X402_BUYER_GUIDE_URL,
    X402Config,
    create_app,
    landing_page_html,
    optional_dependency_help,
    payment_manifest,
    pricing_summary,
    tenant_access_token,
    normalize_memory_backend,
    x402_route_configs,
)
from holographic_product import LocalAgentCore, demo
from lecore import __version__ as LECORE_VERSION


def _memory_keys(active="v1", include_old=False):
    def encoded(key_id):
        return base64.urlsafe_b64encode((key_id * 32).encode("ascii")[:32]).decode("ascii")

    keys = {active: encoded(active)}
    if include_old and active != "v1":
        keys["v1"] = encoded("v1")
    return json.dumps({"active": active, "keys": keys})


def test_default_x402_config_uses_testnet_price_shape():
    cfg = X402Config(pay_to="0xabc")

    assert cfg.network == DEFAULT_NETWORK
    assert cfg.price == DEFAULT_PRICE and cfg.price.startswith("$")
    assert cfg.facilitator_url == "https://x402.org/facilitator"
    assert cfg.public_url == DEFAULT_PUBLIC_URL


def test_payment_manifest_protects_specific_memory_and_compute_routes_only():
    manifest = payment_manifest(X402Config(pay_to="0xabc"))
    routes = {row["route"] for row in manifest}

    assert routes == {
        "DELETE /v1/memory",
        "GET /v1/memory",
        "PATCH /v1/memory",
        "POST /v1/memory",
        "POST /v1/recall",
        "POST /v1/route",
        "GET /v1/dashboard",
    }
    assert all("*" not in route for route in routes)
    assert "POST /admin/remember" not in routes
    assert "POST /admin/tenant-token" not in routes
    assert "GET /health" not in routes
    assert all(row["accepts"][0]["pay_to"] == "0xabc" for row in manifest)
    descriptions = " ".join(row["description"] for row in manifest).lower()
    assert "tenant-scoped agent memory" in descriptions
    assert "localagentcore" not in descriptions
    assert "local agent" not in descriptions


def test_price_validation_keeps_x402_format_honest():
    with pytest.raises(ValueError, match="dollar prefix"):
        X402Config(pay_to="0xabc", price="0.001")
    with pytest.raises(ValueError, match="positive dollar amount"):
        X402Config(pay_to="0xabc", price="$0")


def test_x402_route_configs_build_against_optional_sdk():
    pytest.importorskip("x402")

    routes = x402_route_configs(
        X402Config(pay_to="0xabc", public_url="https://api.example.test/")
    )

    assert sorted(routes) == [
        "DELETE /v1/memory",
        "GET /v1/dashboard",
        "GET /v1/memory",
        "PATCH /v1/memory",
        "POST /v1/memory",
        "POST /v1/recall",
        "POST /v1/route",
    ]
    assert routes["GET /v1/dashboard"].resource == "https://api.example.test/v1/dashboard"
    assert routes["GET /v1/memory"].resource == "https://api.example.test/v1/memory"
    assert routes["PATCH /v1/memory"].resource == "https://api.example.test/v1/memory"
    assert routes["DELETE /v1/memory"].resource == "https://api.example.test/v1/memory"
    assert routes["POST /v1/memory"].resource == "https://api.example.test/v1/memory"
    assert routes["POST /v1/recall"].resource == "https://api.example.test/v1/recall"
    assert routes["POST /v1/route"].resource == "https://api.example.test/v1/route"


def test_env_config_requires_pay_to_for_paid_mode(monkeypatch):
    monkeypatch.delenv("LECORE_X402_PAY_TO", raising=False)

    with pytest.raises(ValueError, match="LECORE_X402_PAY_TO"):
        X402Config.from_env(require_pay_to=True)

    monkeypatch.setenv("LECORE_X402_PUBLIC_URL", "https://api.example.test/")
    config = X402Config.from_env(require_pay_to=False)

    assert config.pay_to == "0xYourAddress"
    assert config.public_url == "https://api.example.test"


@pytest.mark.parametrize(
    "public_url, message",
    [
        ("api.example.test", "absolute http"),
        ("ftp://api.example.test", "absolute http"),
        ("https://", "absolute http"),
        ("https://user:secret@api.example.test", "credentials"),
        ("https://api.example.test?tenant=public", "query or fragment"),
        ("https://api.example.test#pricing", "query or fragment"),
        ("https://api.example.test:bad", "absolute http"),
        ("https://api.example.test:", "absolute http"),
        ("https://.", "absolute http"),
        ("https://api.example.test /base", "absolute http"),
        ("https://api.example.test\\base", "absolute http"),
    ],
)
def test_public_url_rejects_unsafe_or_ambiguous_values(public_url, message):
    with pytest.raises(ValueError, match=message):
        X402Config(pay_to="0xabc", public_url=public_url)


def test_paid_mode_requires_https_outside_localhost(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("x402")

    with pytest.raises(ValueError, match="must use https"):
        create_app(
            config=X402Config(pay_to="0xabc", public_url="http://api.example.test"),
            paid=True,
        )

    local = create_app(
        config=X402Config(pay_to="0xabc", public_url="http://127.0.0.1:4021"),
        paid=True,
        tenant_state_dir=tmp_path,
        memory_keys=_memory_keys(),
    )
    assert local is not None


def test_optional_dependency_help_points_to_extra():
    assert 'pip install ".[x402]"' in optional_dependency_help()


def test_memory_backend_selection_is_explicit():
    assert normalize_memory_backend("core") == "core"
    assert normalize_memory_backend("NoSQLite") == MEMORY_BACKEND_NOSQLITE
    with pytest.raises(ValueError, match="'core' or 'nosqlite'"):
        normalize_memory_backend("sqlite")


def test_memory_keyring_requires_versioned_256_bit_keys():
    keyring = MemoryKeyring.from_json(_memory_keys())
    assert keyring.active == "v1"
    assert len(keyring.keys["v1"]) == 32

    with pytest.raises(ValueError, match="exactly 32 bytes"):
        MemoryKeyring.from_json(json.dumps({"active": "v1", "keys": {"v1": "c2hvcnQ="}}))
    missing_active = json.loads(_memory_keys())
    missing_active["active"] = "v2"
    with pytest.raises(ValueError, match="not present"):
        MemoryKeyring.from_json(json.dumps(missing_active))


def test_memory_state_is_compressed_authenticated_and_context_bound(tmp_path):
    codec = MemoryStateCodec(MemoryKeyring.from_json(_memory_keys()))
    path = tmp_path / "acme.json"
    value = {"entries": [{"text": "private-memory-phrase-" * 5000}], "next_id": 2}

    codec.write_json(path, value, "core:acme")
    envelope = path.read_bytes()

    assert envelope.startswith(b"LECMEM01")
    assert b"private-memory-phrase" not in envelope
    assert len(envelope) < len(json.dumps(value).encode("utf-8"))
    assert path.stat().st_mode & 0o777 == 0o600
    assert codec.read_json(path, "core:acme") == value

    with pytest.raises(MemoryStateError, match="authentication failed"):
        codec.read_json(path, "core:other-tenant")

    tampered = bytearray(envelope)
    tampered[-1] ^= 1
    path.write_bytes(tampered)
    with pytest.raises(MemoryStateError, match="authentication failed"):
        codec.read_json(path, "core:acme")


def test_memory_key_rotation_rewraps_state_under_the_active_key(tmp_path):
    path = tmp_path / "public.json"
    old = MemoryStateCodec(MemoryKeyring.from_json(_memory_keys("v1")))
    old.write_json(path, {"entries": [{"text": "rotate me"}]}, "core:public")
    old_envelope = path.read_bytes()

    rotating = MemoryStateCodec(MemoryKeyring.from_json(_memory_keys("v2", include_old=True)))
    assert rotating.read_json(path, "core:public")["entries"][0]["text"] == "rotate me"
    assert path.read_bytes() != old_envelope

    new_only = MemoryStateCodec(MemoryKeyring.from_json(_memory_keys("v2")))
    assert new_only.read_json(path, "core:public")["entries"][0]["text"] == "rotate me"
    with pytest.raises(MemoryStateError, match="unavailable memory key"):
        old.read_json(path, "core:public")


def test_plaintext_memory_requires_explicit_one_time_migration(tmp_path):
    path = tmp_path / "public.json"
    path.write_text('{"entries": [{"text": "legacy plaintext"}]}', encoding="utf-8")
    keyring = MemoryKeyring.from_json(_memory_keys())

    with pytest.raises(MemoryStateError, match="plaintext durable state"):
        MemoryStateCodec(keyring).read_json(path, "core:public")

    migrating = MemoryStateCodec(keyring, allow_plaintext_migration=True)
    assert migrating.read_json(path, "core:public")["entries"][0]["text"] == "legacy plaintext"
    assert path.read_bytes().startswith(b"LECMEM01")
    assert b"legacy plaintext" not in path.read_bytes()
    assert MemoryStateCodec(keyring).read_json(path, "core:public")["entries"][0]["text"] == "legacy plaintext"


def test_paid_durable_memory_fails_closed_without_encryption_key(tmp_path):
    pytest.importorskip("fastapi")
    with pytest.raises(ValueError, match=MEMORY_KEY_ENV):
        create_app(
            config=X402Config(pay_to="0xabc"),
            paid=True,
            tenant_state_dir=tmp_path,
        )


def test_nosqlite_backend_requires_durable_state_dirs(tmp_path):
    pytest.importorskip("fastapi")

    with pytest.raises(ValueError, match="TENANT_STATE_DIR"):
        create_app(
            config=X402Config(pay_to="0xabc"),
            paid=False,
            memory_backend=MEMORY_BACKEND_NOSQLITE,
        )

    with pytest.raises(ValueError, match="NOSQLITE_DATA_DIR"):
        create_app(
            config=X402Config(pay_to="0xabc"),
            paid=False,
            memory_backend=MEMORY_BACKEND_NOSQLITE,
            tenant_state_dir=tmp_path / "core",
        )

    with pytest.raises(ValueError, match="does not permit the plaintext NoSQLite"):
        create_app(
            config=X402Config(pay_to="0xabc"),
            paid=False,
            memory_backend=MEMORY_BACKEND_NOSQLITE,
            tenant_state_dir=tmp_path / "encrypted-core",
            memory_keys=_memory_keys(),
        )


def _nosqlite_binary() -> str:
    binary = os.environ.get("LECORE_X402_NOSQLITE_BIN")
    if not binary or not Path(binary).is_file():
        pytest.skip("set LECORE_X402_NOSQLITE_BIN to run the optional NoSQLite integration test")
    return binary


def test_landing_page_marks_the_testnet_api_as_a_preview():
    html = landing_page_html(X402Config(pay_to="0x96e1604E92A8A1edD0701be3E67Bd4366e87BB84"))

    assert f"<title>{escape(SERVICE_NAME)}</title>" in html
    assert "Testnet developer preview" in html
    assert f"<h1 id=\"hero-title\">{escape(HERO_TITLE)}</h1>" in html
    assert "$0.0011 per request" in html
    assert "$1.10 per 1,000 requests" in html
    assert "does not accept production payments" in html
    assert "Base Sepolia x402" in html
    assert "/pricing" in html
    assert "/v1/dashboard" in html
    assert 'href="/docs"' in html
    assert 'href="/redoc"' in html
    assert 'href="/openapi.json"' in html
    assert "A hosted HTTPS API" in html
    assert "storing and recalling encrypted private-tenant memory" in html
    assert "/v1/memory" in html
    assert "shared public dataset stays read-only" in html
    assert "operator-issued" in html
    assert "Encrypted before durable memory reaches disk" in html
    assert "ready to integrate" not in html
    assert "memory.entries" not in html
    assert "curl -i %s/v1/dashboard" % DEFAULT_PUBLIC_URL in html
    assert "Payment-Required" in html
    assert "Payment-Signature" in html
    assert "Payment-Response" in html
    assert X402_BUYER_GUIDE_URL in html
    assert ":focus-visible" in html
    assert "prefers-reduced-motion" in html
    assert "min-height:44px" in html
    assert "outline:3px solid currentColor" in html
    assert 'href="/docs#/Paid%20API/getDashboard"' in html
    assert "dashboard_v1_dashboard_get" not in html
    assert 'id="quickstart" class="section quickstart" tabindex="-1"' in html
    assert "--coral-text:#b6402f" in html
    assert DEFAULT_PUBLIC_URL in html
    assert "leOS" not in html
    assert "local agent" not in html.lower()
    assert "local-memory" not in html.lower()


def test_landing_page_uses_the_configured_public_url():
    html = landing_page_html(
        X402Config(pay_to="0xabc", public_url="https://api.example.test/base/")
    )

    assert "https://api.example.test/base" in html


def test_paid_challenge_uses_canonical_resource_not_request_headers(monkeypatch, tmp_path):
    pytest.importorskip("x402")
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from x402 import SupportedKind, SupportedResponse
    from x402.http import HTTPFacilitatorClient, decode_payment_required_header

    monkeypatch.setattr(
        HTTPFacilitatorClient,
        "get_supported",
        lambda _client: SupportedResponse(
            kinds=[
                SupportedKind(
                    x402_version=2,
                    scheme="exact",
                    network=DEFAULT_NETWORK,
                )
            ]
        ),
    )
    client = fastapi_testclient.TestClient(
        create_app(
            config=X402Config(
                pay_to="0x96e1604E92A8A1edD0701be3E67Bd4366e87BB84",
                public_url=DEFAULT_PUBLIC_URL,
            ),
            paid=True,
            tenant_state_dir=tmp_path,
            memory_keys=_memory_keys(),
        )
    )

    response = client.get(
        "/v1/dashboard",
        headers={"host": "attacker.invalid", "x-forwarded-proto": "http"},
    )
    assert response.status_code == 402
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["strict-transport-security"] == "max-age=31536000"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert "unsafe-inline" not in response.headers["content-security-policy"]
    assert "sepolia.base.org" not in response.headers["content-security-policy"]
    challenge = decode_payment_required_header(response.headers["payment-required"])
    assert challenge.resource.url == DEFAULT_PUBLIC_URL + "/v1/dashboard"
    assert challenge.resource.description == "Read the service readiness dashboard"
    assert "LocalAgentCore" not in challenge.resource.description
    memory_challenge_response = client.post(
        "/v1/memory",
        headers={IDEMPOTENCY_HEADER: "challenge-only"},
        json={"tenant": "acme", "text": "not written before payment"},
    )
    assert memory_challenge_response.status_code == 402
    memory_challenge = decode_payment_required_header(
        memory_challenge_response.headers["payment-required"]
    )
    assert memory_challenge.resource.url == DEFAULT_PUBLIC_URL + "/v1/memory"
    assert not (tmp_path / "acme.json").exists()

    browser_response = client.get(
        "/v1/dashboard",
        headers={"accept": "text/html", "user-agent": "Mozilla/5.0"},
    )
    assert browser_response.status_code == 402
    assert browser_response.headers["content-type"].startswith("text/html")
    assert browser_response.headers["cache-control"] == "no-store"
    assert browser_response.headers["x-content-type-options"] == "nosniff"
    paywall_csp = browser_response.headers["content-security-policy"]
    assert "script-src 'unsafe-inline'" in paywall_csp
    assert "style-src 'unsafe-inline'" in paywall_csp
    assert "connect-src 'self' https://sepolia.base.org" in paywall_csp
    assert "https://rpc.wallet.coinbase.com" in paywall_csp
    assert "object-src 'none'" in paywall_csp
    assert "frame-src 'none'" in paywall_csp
    assert "frame-ancestors 'none'" in paywall_csp
    assert '<script type="module">' in browser_response.text
    assert "window.x402" in browser_response.text
    assert 'id="root"' in browser_response.text


def test_pricing_summary_distinguishes_testnet_preview_from_production():
    preview = pricing_summary(X402Config(pay_to="0xabc"))
    production = pricing_summary(X402Config(pay_to="0xabc", network="eip155:8453"))

    assert preview == {
        "environment": "testnet_preview",
        "environment_label": "Testnet developer preview",
        "payment_asset": "testnet USDC",
        "per_request": DEFAULT_PRICE,
        "per_1000_requests": "$1.10",
        "display_price": "$1.10 per 1,000 requests",
        "payment_notice": "This Base Sepolia developer preview uses testnet USDC and does not accept production payments.",
    }
    assert production["environment"] == "production"
    assert production["payment_asset"] == "USDC"
    assert production["payment_notice"] == "Payments settle in USDC through x402."


def test_unpaid_dev_app_serves_landing_page_and_keeps_api_routes_free():
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    client = fastapi_testclient.TestClient(create_app(config=X402Config(pay_to="0xabc"), paid=False))

    assert client.app.version == LECORE_VERSION
    landing = client.get("/")
    assert landing.status_code == 200
    assert landing.headers["content-type"].startswith("text/html")
    assert landing.headers["cache-control"] == "public, max-age=60, must-revalidate"
    assert landing.headers["x-content-type-options"] == "nosniff"
    assert landing.headers["x-frame-options"] == "DENY"
    assert landing.headers["referrer-policy"] == "no-referrer"
    assert landing.headers["permissions-policy"] == "camera=(), microphone=(), geolocation=()"
    assert landing.headers["strict-transport-security"] == "max-age=31536000"
    assert "style-src 'unsafe-inline'" in landing.headers["content-security-policy"]
    assert escape(SERVICE_NAME) in landing.text

    health = client.get("/health")
    assert health.status_code == 200
    assert health.headers["cache-control"] == "no-store"
    assert health.json()["ok"] is True

    pricing = client.get("/pricing")
    assert pricing.status_code == 200
    assert pricing.headers["cache-control"] == "public, max-age=60, must-revalidate"
    assert pricing.json()["x402"]["price"] == DEFAULT_PRICE
    assert pricing.json()["pricing"]["environment"] == "testnet_preview"
    assert pricing.json()["pricing"]["per_1000_requests"] == "$1.10"
    assert pricing.json()["documentation"] == {
        "swagger_ui": DEFAULT_PUBLIC_URL + "/docs",
        "reference": DEFAULT_PUBLIC_URL + "/redoc",
        "openapi_schema": DEFAULT_PUBLIC_URL + "/openapi.json",
    }
    assert pricing.json()["tenancy"]["default_tenant"] == DEFAULT_TENANT_ID
    assert {row["route"] for row in pricing.json()["routes"]} == {
        "DELETE /v1/memory",
        "GET /v1/memory",
        "PATCH /v1/memory",
        "POST /v1/memory",
        "POST /v1/recall",
        "POST /v1/route",
        "GET /v1/dashboard",
    }
    missing = client.get("/leos/v1/dashboard")
    assert missing.status_code == 404
    assert missing.headers["cache-control"] == "no-store"


def test_response_policy_covers_admin_auth_and_local_http():
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    secured = fastapi_testclient.TestClient(
        create_app(
            config=X402Config(pay_to="0xabc"),
            paid=False,
            admin_token="admin-secret",
        )
    )

    unauthorized = secured.post(
        "/admin/remember",
        headers={"X-Admin-Token": "wrong"},
        json={"text": "blocked"},
    )

    assert unauthorized.status_code == 401
    assert unauthorized.headers["cache-control"] == "no-store"
    assert unauthorized.headers["x-content-type-options"] == "nosniff"

    local = fastapi_testclient.TestClient(
        create_app(
            config=X402Config(pay_to="0xabc", public_url="http://localhost:4021"),
            paid=False,
        )
    )
    assert "strict-transport-security" not in local.get("/").headers


def test_public_docs_describe_the_x402_contract_and_hide_operator_routes(tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    pytest.importorskip("x402")
    config = X402Config(pay_to="0xabc", public_url="https://api.example.test")
    client = fastapi_testclient.TestClient(
        create_app(config=config, paid=True, tenant_state_dir=tmp_path, memory_keys=_memory_keys())
    )

    docs = client.get("/docs")
    redoc = client.get("/redoc")
    openapi = client.get("/openapi.json")

    assert docs.status_code == 200
    assert redoc.status_code == 200
    assert openapi.status_code == 200
    assert "/openapi.json" in docs.text
    assert "/openapi.json" in redoc.text
    assert "https://cdn.jsdelivr.net" in docs.headers["content-security-policy"]
    assert "https://fonts.gstatic.com" in redoc.headers["content-security-policy"]

    schema = openapi.json()
    assert schema["info"]["title"] == SERVICE_NAME
    assert schema["info"]["version"] == LECORE_VERSION
    assert "Payment-Signature" in schema["info"]["description"]
    assert schema["servers"] == [{"url": "https://api.example.test", "description": "Public API"}]
    assert schema["externalDocs"] == {
        "description": "x402 buyer quickstart",
        "url": X402_BUYER_GUIDE_URL,
    }
    assert {tag["name"] for tag in schema["tags"]} == {"Discovery", "Paid API"}
    assert "/admin/remember" not in schema["paths"]
    assert "/admin/tenant-token" not in schema["paths"]

    assert {
        (path, method): operation["operationId"]
        for path, operations in schema["paths"].items()
        for method, operation in operations.items()
    } == {
        ("/health", "get"): "getHealth",
        ("/pricing", "get"): "getPricing",
        ("/v1/memory", "post"): "storeMemory",
        ("/v1/memory", "get"): "getMemory",
        ("/v1/memory", "patch"): "updateMemory",
        ("/v1/memory", "delete"): "deleteMemory",
        ("/v1/recall", "post"): "recallMemory",
        ("/v1/route", "post"): "routeTask",
        ("/v1/dashboard", "get"): "getDashboard",
    }

    health_schema = schema["paths"]["/health"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert set(health_schema["required"]) == {"ok", "name", "paid", "memory", "memory_backend", "tenancy"}
    pricing_schema = schema["paths"]["/pricing"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert {"documentation", "x402", "pricing", "routes"}.issubset(pricing_schema["required"])

    manifest = payment_manifest(config)
    for row in manifest:
        method, path = row["route"].lower().split(" ", 1)
        operation = schema["paths"][path][method]
        assert operation["tags"] == ["Paid API"]
        assert {"200", "400", "401", "402", "403", "502"}.issubset(operation["responses"])
        assert "Payment-Response" in operation["responses"]["200"]["headers"]
        assert {"Payment-Required", "Payment-Response"} == set(operation["responses"]["402"]["headers"])
        assert "text/html" in operation["responses"]["402"]["content"]
        assert operation["responses"]["502"]["content"]["application/json"]["schema"]["required"] == ["error"]
        payment_parameter = next(
            parameter for parameter in operation["parameters"]
            if parameter["name"] == "Payment-Signature"
        )
        assert payment_parameter["schema"]["format"] == "byte"
        assert "Omit to receive" in payment_parameter["description"]

    assert "503" in schema["paths"]["/v1/recall"]["post"]["responses"]
    assert "503" not in schema["paths"]["/v1/route"]["post"]["responses"]

    memory_operation = schema["paths"]["/v1/memory"]["post"]
    assert {"409", "402"}.issubset(memory_operation["responses"])
    assert next(
        parameter for parameter in memory_operation["parameters"]
        if parameter["name"] == IDEMPOTENCY_HEADER
    )["required"] is True
    memory_content = memory_operation["requestBody"]["content"]["application/json"]
    assert memory_content["schema"]["required"] == ["text"]
    assert memory_content["schema"]["properties"]["text"]["maxLength"] == 65536
    assert "encrypted" in memory_operation["description"].lower()
    get_memory_operation = schema["paths"]["/v1/memory"]["get"]
    assert {parameter["name"] for parameter in get_memory_operation["parameters"]} >= {
        "memory_id", "limit", "cursor", TENANT_HEADER, TENANT_TOKEN_HEADER, "Payment-Signature"
    }
    assert get_memory_operation["responses"]["200"]["content"]["application/json"]["schema"]["properties"]["items"]["maxItems"] == 100
    assert "404" in get_memory_operation["responses"]
    update_memory_operation = schema["paths"]["/v1/memory"]["patch"]
    update_content = update_memory_operation["requestBody"]["content"]["application/json"]
    assert update_content["schema"]["anyOf"] == [
        {"required": ["text"]},
        {"required": ["label"]},
        {"required": ["metadata"]},
    ]
    assert update_content["schema"]["additionalProperties"] is False
    assert next(
        parameter for parameter in update_memory_operation["parameters"]
        if parameter["name"] == "memory_id"
    )["required"] is True
    assert "404" in update_memory_operation["responses"]
    delete_memory_operation = schema["paths"]["/v1/memory"]["delete"]
    assert next(
        parameter for parameter in delete_memory_operation["parameters"]
        if parameter["name"] == "memory_id"
    )["required"] is True
    assert "idempotently" in delete_memory_operation["description"].lower()

    recall_content = schema["paths"]["/v1/recall"]["post"]["requestBody"]["content"]["application/json"]
    assert recall_content["schema"]["required"] == ["query"]
    assert recall_content["schema"]["properties"]["k"]["maximum"] == 100
    assert recall_content["schema"]["properties"]["query"]["pattern"] == r"\S"
    assert "pattern" not in recall_content["schema"]["properties"]["tenant"]
    assert recall_content["examples"]["public"]["value"]["query"] == "deterministic agent memory"
    recall_success = schema["paths"]["/v1/recall"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert recall_success["required"] == ["ok", "tenant", "query", "hits"]
    assert recall_success["properties"]["hits"]["items"]["required"] == ["id", "text", "label", "metadata", "score"]

    route_content = schema["paths"]["/v1/route"]["post"]["requestBody"]["content"]["application/json"]
    assert route_content["schema"]["required"] == ["task"]
    assert route_content["schema"]["properties"]["task"]["pattern"] == r"\S"
    assert "semantic memory retrieval" in route_content["examples"]["public"]["value"]["task"]
    route_success = schema["paths"]["/v1/route"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert route_success["properties"]["route"]["properties"]["decision"]["enum"] == ["act", "choose", "unknown"]

    dashboard_success = schema["paths"]["/v1/dashboard"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert "self_contained_engine" in dashboard_success["properties"]["dashboard"]["properties"]["checks"]["properties"]

    serialized = json.dumps(schema).lower()
    assert "localagentcore" not in serialized
    assert "local agent" not in serialized

    remember = client.post("/admin/remember", json={"text": "blocked"})
    tenant_token = client.post("/admin/tenant-token", json={"tenant": "acme"})
    assert remember.status_code == 403
    assert tenant_token.status_code == 403
    assert remember.headers["cache-control"] == "no-store"
    assert tenant_token.headers["cache-control"] == "no-store"
    assert remember.headers["x-content-type-options"] == "nosniff"


def test_dashboard_translates_embedded_sdk_terms_at_the_http_boundary():
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    client = fastapi_testclient.TestClient(create_app(config=X402Config(pay_to="0xabc"), paid=False))

    response = client.get("/v1/dashboard")

    assert response.status_code == 200
    dashboard = response.json()["dashboard"]
    assert dashboard["name"] == SERVICE_NAME
    assert dashboard["checks"]["self_contained_engine"] is True
    assert "local_only" not in dashboard["checks"]
    assert "localagentcore" not in json.dumps(dashboard).lower()


def test_health_does_not_run_expensive_evidence_probe():
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    core = demo()

    def fail_evidence():
        raise AssertionError("health must not run evidence")

    core.evidence = fail_evidence
    client = fastapi_testclient.TestClient(
        create_app(core=core, config=X402Config(pay_to="0xabc"), paid=False)
    )

    response = client.get("/health")
    pricing = client.get("/pricing").json()

    assert response.status_code == 200
    assert response.json()["memory"]["entries"] == 3
    assert pricing["pricing"]["environment"] == "testnet_preview"
    assert all(not row["route"].split(" ", 1)[1].startswith("/leos/") for row in pricing["routes"])
    assert client.get("/leos/v1/dashboard").status_code == 404


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "x", "k": "bad"},
        {"query": "x", "k": 0},
        {"query": "x", "k": -1},
        {"query": ""},
        {"query": "x", "abstain": "bad"},
        {"query": "x", "abstain": 1.1},
    ],
)
def test_recall_rejects_invalid_inputs(payload):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    client = fastapi_testclient.TestClient(
        create_app(config=X402Config(pay_to="0xabc"), paid=False)
    )

    response = client.post("/v1/recall", json=payload)

    assert response.status_code == 400


def test_tenant_id_must_be_a_string_and_match_the_header():
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    client = fastapi_testclient.TestClient(
        create_app(
            config=X402Config(pay_to="0xabc"),
            paid=False,
            admin_token="admin-secret",
        )
    )

    numeric = client.post(
        "/admin/remember",
        headers={"X-Admin-Token": "admin-secret"},
        json={"tenant": 0, "text": "must not reach public"},
    )
    mismatch = client.post(
        "/admin/remember",
        headers={"X-Admin-Token": "admin-secret", TENANT_HEADER: "acme"},
        json={"tenant": "beta", "text": "must not cross tenants"},
    )

    assert numeric.status_code == 400
    assert mismatch.status_code == 400


def test_private_tenant_memory_requires_a_tenant_token():
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    client = fastapi_testclient.TestClient(
        create_app(
            config=X402Config(pay_to="0xabc"),
            paid=False,
            admin_token="admin-secret",
            tenant_secret="tenant-secret",
        )
    )

    issued = client.post(
        "/admin/tenant-token",
        headers={"X-Admin-Token": "admin-secret"},
        json={"tenant": "acme"},
    )
    assert issued.status_code == 200
    tenant_token = issued.json()["tenant_token"]
    assert tenant_token == tenant_access_token("acme", "tenant-secret")

    written = client.post(
        "/admin/remember",
        headers={"X-Admin-Token": "admin-secret", TENANT_HEADER: "acme"},
        json={"text": "acme-private-omega memory", "label": "tenant-memory"},
    )
    assert written.status_code == 200
    assert written.json()["tenant"] == "acme"

    blocked = client.post(
        "/v1/recall",
        headers={TENANT_HEADER: "acme"},
        json={"query": "acme private omega"},
    )
    assert blocked.status_code == 401

    recalled = client.post(
        "/v1/recall",
        headers={TENANT_HEADER: "acme", TENANT_TOKEN_HEADER: tenant_token},
        json={"query": "acme private omega"},
    )
    assert recalled.status_code == 200
    assert recalled.json()["tenant"] == "acme"
    assert recalled.json()["hits"][0]["label"] == "tenant-memory"

    public_recall = client.post("/v1/recall", json={"query": "acme private omega"})
    assert public_recall.status_code == 200
    assert public_recall.json()["tenant"] == DEFAULT_TENANT_ID
    assert all(hit["label"] != "tenant-memory" for hit in public_recall.json()["hits"])


def test_tenant_memory_can_persist_to_state_dir(tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    first = fastapi_testclient.TestClient(
        create_app(
            config=X402Config(pay_to="0xabc"),
            paid=False,
            admin_token="admin-secret",
            tenant_secret="tenant-secret",
            tenant_state_dir=tmp_path,
        )
    )
    first.post(
        "/admin/remember",
        headers={"X-Admin-Token": "admin-secret"},
        json={"tenant": "acme", "text": "persisted tenant recall text", "label": "persisted"},
    )

    second = fastapi_testclient.TestClient(
        create_app(
            config=X402Config(pay_to="0xabc"),
            paid=False,
            tenant_secret="tenant-secret",
            tenant_state_dir=tmp_path,
        )
    )
    recalled = second.post(
        "/v1/recall",
        headers={
            TENANT_HEADER: "acme",
            TENANT_TOKEN_HEADER: tenant_access_token("acme", "tenant-secret"),
        },
        json={"query": "persisted tenant recall"},
    )

    assert recalled.status_code == 200
    assert recalled.json()["hits"][0]["label"] == "persisted"


def test_tenant_memory_and_write_journal_are_encrypted_on_disk(tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    kwargs = {
        "config": X402Config(pay_to="0xabc"),
        "paid": False,
        "tenant_secret": "tenant-secret",
        "tenant_state_dir": tmp_path,
        "memory_keys": _memory_keys(),
    }
    with fastapi_testclient.TestClient(create_app(admin_token="admin-secret", **kwargs)) as first:
        written = first.post(
            "/admin/remember",
            headers={"X-Admin-Token": "admin-secret", IDEMPOTENCY_HEADER: "encrypted-001"},
            json={"tenant": "acme", "text": "encrypted tenant recall phrase", "label": "encrypted"},
        )
        assert written.status_code == 200
        storage = first.get("/health").json()["memory_backend"]["storage"]
        assert storage == {
            "durable": True,
            "encrypted": True,
            "cipher": MEMORY_CIPHER,
            "compression": MEMORY_COMPRESSION,
            "plaintext_migration_enabled": False,
        }

    files = [tmp_path / "acme.json"] + list((tmp_path / ".x402-memory-transactions").glob("*/*.json"))
    assert len(files) == 2
    for path in files:
        envelope = path.read_bytes()
        assert envelope.startswith(b"LECMEM01")
        assert b"encrypted tenant recall phrase" not in envelope
        assert path.stat().st_mode & 0o777 == 0o600

    with fastapi_testclient.TestClient(create_app(**kwargs)) as restarted:
        recalled = restarted.post(
            "/v1/recall",
            headers={
                TENANT_HEADER: "acme",
                TENANT_TOKEN_HEADER: tenant_access_token("acme", "tenant-secret"),
            },
            json={"query": "encrypted tenant recall"},
        )
        assert recalled.status_code == 200
        assert recalled.json()["hits"][0]["label"] == "encrypted"


def test_one_time_migration_rewrites_legacy_core_and_journal_then_fails_closed(tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    token = tenant_access_token("acme", "tenant-secret")
    common = {
        "config": X402Config(pay_to="0xabc"),
        "paid": False,
        "tenant_secret": "tenant-secret",
        "tenant_state_dir": tmp_path,
    }
    with fastapi_testclient.TestClient(create_app(admin_token="admin-secret", **common)) as legacy:
        response = legacy.post(
            "/admin/remember",
            headers={"X-Admin-Token": "admin-secret", IDEMPOTENCY_HEADER: "legacy-001"},
            json={"tenant": "acme", "text": "legacy state migration phrase", "label": "migrated"},
        )
        assert response.status_code == 200

    durable_files = [tmp_path / "acme.json"] + list((tmp_path / ".x402-memory-transactions").glob("*/*.json"))
    assert len(durable_files) == 2
    assert all(b"legacy state migration phrase" in path.read_bytes() for path in durable_files)

    with fastapi_testclient.TestClient(
        create_app(
            **common,
            memory_keys=_memory_keys(),
            allow_plaintext_migration=True,
        )
    ) as migrating:
        recalled = migrating.post(
            "/v1/recall",
            headers={TENANT_HEADER: "acme", TENANT_TOKEN_HEADER: token},
            json={"query": "legacy migration"},
        )
        assert recalled.status_code == 200
        assert recalled.json()["hits"][0]["label"] == "migrated"

    for path in durable_files:
        assert path.read_bytes().startswith(b"LECMEM01")
        assert b"legacy state migration phrase" not in path.read_bytes()

    with fastapi_testclient.TestClient(create_app(**common, memory_keys=_memory_keys())) as strict:
        assert strict.get("/health").json()["memory_backend"]["storage"]["plaintext_migration_enabled"] is False


def test_private_tenant_can_store_list_get_update_delete_and_recall_memory(tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    token = tenant_access_token("acme", "tenant-secret")
    client = fastapi_testclient.TestClient(
        create_app(
            config=X402Config(pay_to="0xabc"),
            paid=False,
            tenant_secret="tenant-secret",
            tenant_state_dir=tmp_path,
            memory_keys=_memory_keys(),
        )
    )
    headers = {
        TENANT_HEADER: "acme",
        TENANT_TOKEN_HEADER: token,
        IDEMPOTENCY_HEADER: "buyer-memory-001",
    }
    auth = {TENANT_HEADER: "acme", TENANT_TOKEN_HEADER: token}
    payload = {
        "text": "buyer-owned encrypted memory phrase",
        "label": "buyer",
        "metadata": {"session": "s1"},
    }

    first = client.post("/v1/memory", headers=headers, json=payload)
    retry = client.post("/v1/memory", headers=headers, json=payload)
    conflict = client.post("/v1/memory", headers=headers, json={"text": "different"})
    second = client.post(
        "/v1/memory",
        headers={**auth, IDEMPOTENCY_HEADER: "buyer-memory-002"},
        json={"text": "second paginated memory", "label": "second"},
    )
    public_write = client.post(
        "/v1/memory",
        headers={IDEMPOTENCY_HEADER: "public-write"},
        json={"text": "must not poison public memory"},
    )
    first_id = first.json()["memory"]["id"]
    second_id = second.json()["memory"]["id"]
    page_one = client.get("/v1/memory", headers=auth, params={"limit": 1})
    page_two = client.get(
        "/v1/memory",
        headers=auth,
        params={"limit": 1, "cursor": page_one.json()["next_cursor"]},
    )
    exact = client.get("/v1/memory", headers=auth, params={"memory_id": second_id})
    public_list = client.get("/v1/memory")

    before_noop_update = (tmp_path / "acme.json").read_bytes()
    noop_update = client.patch(
        "/v1/memory",
        headers=auth,
        params={"memory_id": second_id},
        json={"text": "second paginated memory", "label": "second"},
    )
    assert (tmp_path / "acme.json").read_bytes() == before_noop_update
    updated = client.patch(
        "/v1/memory",
        headers=auth,
        params={"memory_id": second_id},
        json={
            "text": "second memory updated for durable recall",
            "label": None,
            "metadata": {"revision": 2},
        },
    )
    empty_update = client.patch(
        "/v1/memory", headers=auth, params={"memory_id": second_id}, json={}
    )
    unknown_update = client.patch(
        "/v1/memory", headers=auth, params={"memory_id": second_id}, json={"owner": "other"}
    )
    missing_update = client.patch(
        "/v1/memory", headers=auth, params={"memory_id": "missing"}, json={"label": "lost"}
    )
    original_store_after_update = client.post(
        "/v1/memory",
        headers={**auth, IDEMPOTENCY_HEADER: "buyer-memory-002"},
        json={"text": "second paginated memory", "label": "second"},
    )
    updated_exact = client.get("/v1/memory", headers=auth, params={"memory_id": second_id})
    updated_recall = client.post(
        "/v1/recall", headers=auth, json={"query": "updated durable recall"}
    )

    state_path = tmp_path / "acme.json"
    before_absent_delete = state_path.read_bytes()
    absent_delete = client.delete("/v1/memory", headers=auth, params={"memory_id": "missing"})
    assert state_path.read_bytes() == before_absent_delete

    deleted = client.delete("/v1/memory", headers=auth, params={"memory_id": first_id})
    deleted_retry = client.delete("/v1/memory", headers=auth, params={"memory_id": first_id})
    deleted_get = client.get("/v1/memory", headers=auth, params={"memory_id": first_id})
    deleted_store_retry = client.post("/v1/memory", headers=headers, json=payload)
    recalled = client.post(
        "/v1/recall",
        headers=auth,
        json={"query": "buyer encrypted phrase"},
    )

    assert first.status_code == 200 and retry.status_code == 200
    assert first.json()["memory"] == retry.json()["memory"]
    assert first.json()["transaction"]["idempotent"] is True
    assert second.status_code == 200
    assert conflict.status_code == 409
    assert public_write.status_code == 403
    assert public_list.status_code == 422
    assert [item["id"] for item in page_one.json()["items"]] == [first_id]
    assert page_one.json()["next_cursor"] == first_id
    assert [item["id"] for item in page_two.json()["items"]] == [second_id]
    assert page_two.json()["next_cursor"] is None
    assert [item["id"] for item in exact.json()["items"]] == [second_id]
    assert noop_update.status_code == 200
    assert updated.status_code == 200
    assert updated.json()["memory"] == {
        "id": second_id,
        "text": "second memory updated for durable recall",
        "label": None,
        "metadata": {"revision": 2},
    }
    assert empty_update.status_code == 400
    assert unknown_update.status_code == 400
    assert missing_update.status_code == 404
    assert original_store_after_update.status_code == 409
    assert updated_exact.json()["items"] == [updated.json()["memory"]]
    assert updated_recall.json()["hits"][0]["id"] == second_id
    assert absent_delete.json()["deleted"] is False
    assert deleted.json()["deleted"] is True
    assert deleted_retry.json()["deleted"] is False
    assert deleted_get.status_code == 404
    assert deleted_store_retry.status_code == 409
    assert recalled.status_code == 200
    assert all(hit["id"] != first_id for hit in recalled.json()["hits"])

    restarted = fastapi_testclient.TestClient(
        create_app(
            config=X402Config(pay_to="0xabc"),
            paid=False,
            tenant_secret="tenant-secret",
            tenant_state_dir=tmp_path,
            memory_keys=_memory_keys(),
        )
    )
    assert restarted.get("/v1/memory", headers=auth, params={"memory_id": first_id}).status_code == 404
    restarted_second = restarted.get("/v1/memory", headers=auth, params={"memory_id": second_id})
    assert restarted_second.status_code == 200
    assert restarted_second.json()["items"] == [updated.json()["memory"]]


def test_public_memory_persists_across_app_restart(tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    first = fastapi_testclient.TestClient(
        create_app(
            config=X402Config(pay_to="0xabc"),
            paid=False,
            admin_token="admin-secret",
            tenant_state_dir=tmp_path,
        )
    )
    written = first.post(
        "/admin/remember",
        headers={"X-Admin-Token": "admin-secret"},
        json={"text": "unique public persisted phrase", "label": "public-persisted"},
    )
    assert written.status_code == 200

    second = fastapi_testclient.TestClient(
        create_app(
            config=X402Config(pay_to="0xabc"),
            paid=False,
            tenant_state_dir=tmp_path,
        )
    )
    recalled = second.post(
        "/v1/recall",
        json={"query": "unique public persisted phrase", "k": 10},
    )

    assert recalled.status_code == 200
    assert "public-persisted" in [hit["label"] for hit in recalled.json()["hits"]]


def test_durable_memory_transaction_reuses_one_memory_for_retries(tmp_path):
    store = TenantCoreStore(LocalAgentCore(), tmp_path)
    transactions = TenantMemoryTransactions(store, tmp_path)

    first = transactions.remember(
        "acme",
        "one durable transaction memory",
        "journal",
        {"source": "test"},
        "retry-001",
        None,
    )
    second = transactions.remember(
        "acme",
        "one durable transaction memory",
        "journal",
        {"source": "test"},
        "retry-001",
        None,
    )

    assert first["memory"] == second["memory"]
    assert first["transaction"]["state"] == "complete"
    entries = store.read("acme", lambda core: core.entries)
    assert [entry.id for entry in entries] == [first["memory"]["id"]]

    with pytest.raises(MemoryTransactionConflict, match="different memory write"):
        transactions.remember("acme", "different payload", "journal", {}, "retry-001", None)


def test_durable_memory_transaction_recovers_a_failed_mirror(tmp_path):
    class FlakyMirror:
        def __init__(self):
            self.fail = True
            self.memories = []

        def remember(self, tenant_id, memory):
            if self.fail:
                raise NoSQLiteError("mirror offline")
            self.memories.append((tenant_id, dict(memory)))

    store = TenantCoreStore(LocalAgentCore(), tmp_path)
    transactions = TenantMemoryTransactions(store, tmp_path)
    mirror = FlakyMirror()

    with pytest.raises(NoSQLiteError, match="mirror offline") as failed:
        transactions.remember("acme", "recover this mirror write", "journal", {}, "retry-002", mirror)

    committed = store.read("acme", lambda core: [entry.to_dict() for entry in core.entries])
    assert len(committed) == 1
    assert isinstance(failed.value, MemoryMirrorPending)
    pending = transactions.resume("acme", failed.value.transaction_id, None)
    assert pending["transaction"]["state"] == "core_committed"
    assert len(store.read("acme", lambda core: core.entries)) == 1

    restarted = TenantMemoryTransactions(TenantCoreStore(LocalAgentCore(), tmp_path), tmp_path)
    mirror.fail = False
    recovery = restarted.recover_pending(mirror)

    assert recovery == {"recovered": 1, "pending": 0, "invalid": 0}
    assert mirror.memories == [("acme", committed[0])]
    retried = restarted.remember("acme", "recover this mirror write", "journal", {}, "retry-002", mirror)
    assert retried["memory"] == committed[0]
    assert len(TenantCoreStore(LocalAgentCore(), tmp_path).read("acme", lambda core: core.entries)) == 1


def test_nosqlite_projection_replaces_deletes_and_reconciles_exact_snapshot():
    class RecordingProcess:
        generation = 1

        def __init__(self):
            self.commands = []

        def ensure_started(self):
            return self.generation

        def command(self, command):
            self.commands.append(command)
            return {"ok": True}

    process = RecordingProcess()
    store = NoSQLiteMemoryStore.__new__(NoSQLiteMemoryStore)
    store._dimensions = 128
    store._process = process
    store._lock = threading.RLock()
    store._encoder_generation = 1
    collection = store._collection_name("acme")
    store._ready_collections = {collection}
    store._synced_collections = {(1, collection)}
    memory = {"id": "m1", "text": "updated", "label": None, "metadata": {"v": 2}}

    store.replace("acme", memory)
    store.delete("acme", "m1")
    store._synced_collections.clear()
    store.sync("acme", [memory])

    assert process.commands[0] == {"delete": collection, "filter": {"_id": "m1"}}
    assert process.commands[1]["insert"] == collection
    assert process.commands[1]["documents"][0]["text"] == "updated"
    assert process.commands[2] == {"delete": collection, "filter": {"_id": "m1"}}
    assert process.commands[3] == {"delete": collection}
    assert process.commands[4]["documents"][0]["_id"] == "m1"
    assert store._synced_collections == {(1, collection)}


def test_admin_remember_idempotency_header_is_durable_and_conflicts_cleanly(tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    client = fastapi_testclient.TestClient(
        create_app(
            config=X402Config(pay_to="0xabc"),
            paid=False,
            admin_token="admin-secret",
            tenant_state_dir=tmp_path,
        )
    )
    headers = {"X-Admin-Token": "admin-secret", IDEMPOTENCY_HEADER: "api-retry-001"}
    payload = {"text": "idempotent API memory", "label": "idempotent"}

    first = client.post("/admin/remember", headers=headers, json=payload)
    second = client.post("/admin/remember", headers=headers, json=payload)
    conflict = client.post(
        "/admin/remember",
        headers=headers,
        json={"text": "idempotent API memory but changed", "label": "idempotent"},
    )

    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["memory"] == second.json()["memory"]
    assert first.json()["transaction"]["state"] == "complete"
    assert conflict.status_code == 409


def test_idempotency_header_requires_a_durable_tenant_state_dir():
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    client = fastapi_testclient.TestClient(
        create_app(config=X402Config(pay_to="0xabc"), paid=False, admin_token="admin-secret")
    )

    response = client.post(
        "/admin/remember",
        headers={"X-Admin-Token": "admin-secret", IDEMPOTENCY_HEADER: "requires-state"},
        json={"text": "this key needs durable state"},
    )

    assert response.status_code == 400
    assert "TENANT_STATE_DIR" in response.json()["detail"]


def test_shadow_mirror_failure_keeps_the_original_unkeyed_transaction(tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    client = fastapi_testclient.TestClient(
        create_app(
            core=LocalAgentCore(),
            config=X402Config(pay_to="0xabc"),
            paid=False,
            admin_token="admin-secret",
            tenant_state_dir=tmp_path / "core",
            nosqlite_shadow=True,
            nosqlite_binary=str(tmp_path / "missing-nosqlite"),
            nosqlite_data_dir=tmp_path / "nosqlite",
        )
    )

    response = client.post(
        "/admin/remember",
        headers={"X-Admin-Token": "admin-secret"},
        json={"text": "shadow write survives its first mirror failure", "label": "shadow"},
    )

    assert response.status_code == 200
    assert response.json()["transaction"]["state"] == "core_committed"
    persisted = TenantCoreStore(LocalAgentCore(), tmp_path / "core")
    entries = persisted.read(DEFAULT_TENANT_ID, lambda core: core.entries)
    assert len(entries) == 1 and entries[0].label == "shadow"


def test_nosqlite_memory_backend_isolates_tenants_and_restarts(tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    binary = _nosqlite_binary()
    data_dir = tmp_path / "nosqlite"
    tenant_token = tenant_access_token("acme", "tenant-secret")
    common = {
        "config": X402Config(pay_to="0xabc"),
        "paid": False,
        "admin_token": "admin-secret",
        "tenant_secret": "tenant-secret",
        "tenant_state_dir": tmp_path / "core",
        "memory_backend": MEMORY_BACKEND_NOSQLITE,
        "nosqlite_binary": binary,
        "nosqlite_data_dir": data_dir,
    }

    with fastapi_testclient.TestClient(create_app(**common)) as first:
        health = first.get("/health")
        assert health.status_code == 200
        assert health.json()["memory_backend"] == {
            "backend": MEMORY_BACKEND_NOSQLITE,
            "nosqlite_shadow": False,
            "nosqlite_configured": True,
            "durable_transactions": True,
            "storage": {
                "durable": True,
                "encrypted": False,
                "cipher": None,
                "compression": None,
                "plaintext_migration_enabled": False,
            },
        }

        public = first.post(
            "/admin/remember",
            headers={"X-Admin-Token": "admin-secret"},
            json={"text": "unique public nosqlite comet memory", "label": "public-nosqlite"},
        )
        private = first.post(
            "/admin/remember",
            headers={"X-Admin-Token": "admin-secret", TENANT_HEADER: "acme"},
            json={"text": "unique acme nosqlite lighthouse memory", "label": "private-nosqlite"},
        )
        assert public.status_code == 200
        assert private.status_code == 200

        acme = first.post(
            "/v1/recall",
            headers={TENANT_HEADER: "acme", TENANT_TOKEN_HEADER: tenant_token},
            json={"query": "acme lighthouse", "k": 10},
        )
        public_recall = first.post(
            "/v1/recall",
            json={"query": "acme lighthouse", "k": 10},
        )
        assert acme.status_code == 200
        assert public_recall.status_code == 200
        assert [hit["label"] for hit in acme.json()["hits"]] == ["private-nosqlite"]
        assert "private-nosqlite" not in [hit["label"] for hit in public_recall.json()["hits"]]

    restart_common = dict(common)
    restart_common.pop("admin_token")
    with fastapi_testclient.TestClient(create_app(**restart_common)) as second:
        persisted = second.post(
            "/v1/recall",
            json={"query": "public comet", "k": 10},
        )
        assert persisted.status_code == 200
        assert "public-nosqlite" in [hit["label"] for hit in persisted.json()["hits"]]


def test_persisted_writes_reload_under_process_lock(tmp_path):
    first = TenantCoreStore(LocalAgentCore(), tmp_path)
    second = TenantCoreStore(LocalAgentCore(), tmp_path)
    first.read("acme", lambda core: core.entries)
    second.read("acme", lambda core: core.entries)

    first.write("acme", lambda core: core.remember("first writer", label="first"))
    second.write("acme", lambda core: core.remember("second writer", label="second"))

    reloaded = TenantCoreStore(LocalAgentCore(), tmp_path)
    labels = [entry.label for entry in reloaded.read("acme", lambda core: core.entries)]
    assert labels == ["first", "second"]


def test_current_cached_tenant_avoids_redecrypt_but_external_changes_reload(tmp_path, monkeypatch):
    codec = MemoryStateCodec(MemoryKeyring.from_json(_memory_keys()))
    first = TenantCoreStore(LocalAgentCore(), tmp_path, codec=codec)
    first.write("acme", lambda core: core.remember("first", label="first"))

    original_read = codec.read_json
    reads = []

    def counted_read(path, context):
        reads.append((path.name, context))
        return original_read(path, context)

    monkeypatch.setattr(codec, "read_json", counted_read)
    first.write("acme", lambda core: core.remember("cached", label="cached"))
    assert reads == []

    second = TenantCoreStore(LocalAgentCore(), tmp_path, codec=codec)
    second.write("acme", lambda core: core.remember("external", label="external"))
    reads.clear()
    first.write("acme", lambda core: core.remember("reloaded", label="reloaded"))

    assert reads == [("acme.json", "core:acme")]
    labels = [entry.label for entry in first.read("acme", lambda core: core.entries)]
    assert labels == ["first", "cached", "external", "reloaded"]
