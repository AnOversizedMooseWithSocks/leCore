"""Tests for the optional x402-paid API publisher."""

import json
from html import escape
import os
from pathlib import Path

import pytest

from holographic_x402_api import (
    DEFAULT_NETWORK,
    DEFAULT_PRICE,
    DEFAULT_PUBLIC_URL,
    DEFAULT_TENANT_ID,
    IDEMPOTENCY_HEADER,
    MEMORY_BACKEND_NOSQLITE,
    MemoryTransactionConflict,
    MemoryMirrorPending,
    NoSQLiteError,
    SERVICE_NAME,
    TENANT_HEADER,
    TENANT_TOKEN_HEADER,
    TenantCoreStore,
    TenantMemoryTransactions,
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


def test_default_x402_config_uses_testnet_price_shape():
    cfg = X402Config(pay_to="0xabc")

    assert cfg.network == DEFAULT_NETWORK
    assert cfg.price == DEFAULT_PRICE and cfg.price.startswith("$")
    assert cfg.facilitator_url == "https://x402.org/facilitator"
    assert cfg.public_url == DEFAULT_PUBLIC_URL


def test_payment_manifest_protects_specific_read_routes_only():
    manifest = payment_manifest(X402Config(pay_to="0xabc"))
    routes = {row["route"] for row in manifest}

    assert routes == {
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
        "GET /v1/dashboard",
        "POST /v1/recall",
        "POST /v1/route",
    ]
    assert routes["GET /v1/dashboard"].resource == "https://api.example.test/v1/dashboard"
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


def test_paid_mode_requires_https_outside_localhost():
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
    )
    assert local is not None


def test_optional_dependency_help_points_to_extra():
    assert 'pip install ".[x402]"' in optional_dependency_help()


def test_memory_backend_selection_is_explicit():
    assert normalize_memory_backend("core") == "core"
    assert normalize_memory_backend("NoSQLite") == MEMORY_BACKEND_NOSQLITE
    with pytest.raises(ValueError, match="'core' or 'nosqlite'"):
        normalize_memory_backend("sqlite")


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


def _nosqlite_binary() -> str:
    binary = os.environ.get("LECORE_X402_NOSQLITE_BIN")
    if not binary or not Path(binary).is_file():
        pytest.skip("set LECORE_X402_NOSQLITE_BIN to run the optional NoSQLite integration test")
    return binary


def test_landing_page_marks_the_testnet_api_as_a_preview():
    html = landing_page_html(X402Config(pay_to="0x96e1604E92A8A1edD0701be3E67Bd4366e87BB84"))

    assert f"<title>{escape(SERVICE_NAME)}</title>" in html
    assert "Testnet developer preview" in html
    assert "$1.10 per 1,000 requests" in html
    assert "does not accept production payments" in html
    assert "Base Sepolia x402" in html
    assert "/pricing" in html
    assert "/v1/dashboard" in html
    assert 'href="/docs"' in html
    assert 'href="/redoc"' in html
    assert 'href="/openapi.json"' in html
    assert "hosted, tenant-scoped agent memory" in html
    assert "0x96e1...BB84" in html
    assert DEFAULT_PUBLIC_URL in html
    assert "leOS" not in html
    assert "local agent" not in html.lower()
    assert "local-memory" not in html.lower()


def test_landing_page_uses_the_configured_public_url():
    html = landing_page_html(
        X402Config(pay_to="0xabc", public_url="https://api.example.test/base/")
    )

    assert "https://api.example.test/base" in html


def test_paid_challenge_uses_canonical_resource_not_request_headers(monkeypatch):
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
        )
    )

    response = client.get(
        "/v1/dashboard",
        headers={"host": "attacker.invalid", "x-forwarded-proto": "http"},
    )
    assert response.status_code == 402
    challenge = decode_payment_required_header(response.headers["payment-required"])
    assert challenge.resource.url == DEFAULT_PUBLIC_URL + "/v1/dashboard"
    assert challenge.resource.description == "Read the service readiness dashboard"
    assert "LocalAgentCore" not in challenge.resource.description


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
    assert escape(SERVICE_NAME) in landing.text

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True

    pricing = client.get("/pricing")
    assert pricing.status_code == 200
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
        "POST /v1/recall",
        "POST /v1/route",
        "GET /v1/dashboard",
    }
    assert client.get("/leos/v1/dashboard").status_code == 404


def test_public_docs_describe_the_x402_contract_and_hide_operator_routes():
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    config = X402Config(pay_to="0xabc", public_url="https://api.example.test")
    client = fastapi_testclient.TestClient(create_app(config=config, paid=False))

    docs = client.get("/docs")
    redoc = client.get("/redoc")
    openapi = client.get("/openapi.json")

    assert docs.status_code == 200
    assert redoc.status_code == 200
    assert openapi.status_code == 200
    assert "/openapi.json" in docs.text
    assert "/openapi.json" in redoc.text

    schema = openapi.json()
    assert schema["info"]["title"] == SERVICE_NAME
    assert schema["info"]["version"] == LECORE_VERSION
    assert "Payment-Signature" in schema["info"]["description"]
    assert schema["servers"] == [{"url": "https://api.example.test", "description": "Public API"}]
    assert {tag["name"] for tag in schema["tags"]} == {"Discovery", "Paid API"}
    assert "/admin/remember" not in schema["paths"]
    assert "/admin/tenant-token" not in schema["paths"]

    manifest = payment_manifest(config)
    for row in manifest:
        method, path = row["route"].lower().split(" ", 1)
        operation = schema["paths"][path][method]
        assert operation["tags"] == ["Paid API"]
        assert "402" in operation["responses"]
        assert "Payment-Required" in operation["responses"]["402"]["headers"]
        assert any(parameter["name"] == "Payment-Signature" for parameter in operation["parameters"])

    recall_content = schema["paths"]["/v1/recall"]["post"]["requestBody"]["content"]["application/json"]
    assert recall_content["schema"]["required"] == ["query"]
    assert recall_content["schema"]["properties"]["k"]["maximum"] == 100
    assert recall_content["examples"]["public"]["value"]["query"] == "deterministic agent memory"

    route_content = schema["paths"]["/v1/route"]["post"]["requestBody"]["content"]["application/json"]
    assert route_content["schema"]["required"] == ["task"]
    assert "semantic memory retrieval" in route_content["examples"]["public"]["value"]["task"]

    serialized = json.dumps(schema).lower()
    assert "localagentcore" not in serialized
    assert "local agent" not in serialized

    assert client.post("/admin/remember", json={"text": "blocked"}).status_code == 403
    assert client.post("/admin/tenant-token", json={"tenant": "acme"}).status_code == 403


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
