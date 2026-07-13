"""Tests for the optional x402-paid API publisher."""

import os
from pathlib import Path

import pytest

from holographic_x402_api import (
    DEFAULT_NETWORK,
    DEFAULT_PRICE,
    DEFAULT_TENANT_ID,
    IDEMPOTENCY_HEADER,
    LEOS_SITE_URL,
    LEOS_ACCESS_HEADER,
    LEOS_TOKEN_CA,
    LEOS_TOKEN_PRICE,
    MEMORY_BACKEND_NOSQLITE,
    MemoryTransactionConflict,
    MemoryMirrorPending,
    NoSQLiteError,
    TENANT_HEADER,
    TENANT_TOKEN_HEADER,
    TenantCoreStore,
    TenantMemoryTransactions,
    X402Config,
    create_app,
    landing_page_html,
    leos_token_offer,
    optional_dependency_help,
    payment_manifest,
    tenant_access_token,
    normalize_memory_backend,
    x402_route_configs,
)
from holographic_product import LocalAgentCore, demo


def test_default_x402_config_uses_testnet_price_shape():
    cfg = X402Config(pay_to="0xabc")

    assert cfg.network == DEFAULT_NETWORK
    assert cfg.price == DEFAULT_PRICE and cfg.price.startswith("$")
    assert cfg.facilitator_url == "https://x402.org/facilitator"


def test_payment_manifest_protects_specific_read_routes_only():
    manifest = payment_manifest(X402Config(pay_to="0xabc"))
    routes = {row["route"] for row in manifest}

    assert routes == {
        "POST /v1/recall",
        "POST /v1/route",
        "GET /v1/dashboard",
        "POST /leos/v1/recall",
        "POST /leos/v1/route",
        "GET /leos/v1/dashboard",
    }
    assert all("*" not in route for route in routes)
    assert "POST /admin/remember" not in routes
    assert "POST /admin/tenant-token" not in routes
    assert "GET /health" not in routes
    assert all(row["accepts"][0]["pay_to"] == "0xabc" for row in manifest)
    assert {
        row["route"]: row["accepts"][0]["price"]
        for row in manifest
        if row["route"].startswith("POST /leos") or row["route"].startswith("GET /leos")
    } == {
        "POST /leos/v1/recall": LEOS_TOKEN_PRICE,
        "POST /leos/v1/route": LEOS_TOKEN_PRICE,
        "GET /leos/v1/dashboard": LEOS_TOKEN_PRICE,
    }


def test_price_validation_keeps_x402_format_honest():
    with pytest.raises(ValueError, match="dollar prefix"):
        X402Config(pay_to="0xabc", price="0.001")


def test_x402_route_configs_build_against_optional_sdk():
    pytest.importorskip("x402")

    routes = x402_route_configs(X402Config(pay_to="0xabc"))

    assert sorted(routes) == [
        "GET /leos/v1/dashboard",
        "GET /v1/dashboard",
        "POST /leos/v1/recall",
        "POST /leos/v1/route",
        "POST /v1/recall",
        "POST /v1/route",
    ]


def test_env_config_requires_pay_to_for_paid_mode(monkeypatch):
    monkeypatch.delenv("LECORE_X402_PAY_TO", raising=False)

    with pytest.raises(ValueError, match="LECORE_X402_PAY_TO"):
        X402Config.from_env(require_pay_to=True)

    assert X402Config.from_env(require_pay_to=False).pay_to == "0xYourAddress"


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


def test_landing_page_explains_why_to_buy_the_api():
    html = landing_page_html(X402Config(pay_to="0x96e1604E92A8A1edD0701be3E67Bd4366e87BB84"))

    assert "<title>leCore x402 API</title>" in html
    assert "Buy the small, useful surface of leCore" in html
    assert "%s per call" % DEFAULT_PRICE in html
    assert LEOS_TOKEN_PRICE in html
    assert LEOS_TOKEN_CA in html
    assert LEOS_SITE_URL in html
    assert "Base Sepolia x402" in html
    assert "/pricing" in html
    assert "/v1/dashboard" in html
    assert "0x96e1...BB84" in html


def test_leos_token_offer_identifies_ca_and_requires_access():
    offer = leos_token_offer()

    assert offer["site"] == LEOS_SITE_URL
    assert offer["ca"] == LEOS_TOKEN_CA
    assert offer["price"] == LEOS_TOKEN_PRICE
    assert "POST /leos/v1/recall" in offer["discount_routes"]
    assert offer["access_header"] == LEOS_ACCESS_HEADER
    assert offer["access_required"] is True
    assert "eligible buyers" in offer["note"]


def test_unpaid_dev_app_serves_landing_page_and_keeps_api_routes_free():
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    client = fastapi_testclient.TestClient(
        create_app(config=X402Config(pay_to="0xabc"), paid=False, leos_access_token="leos-secret")
    )

    landing = client.get("/")
    assert landing.status_code == 200
    assert landing.headers["content-type"].startswith("text/html")
    assert "leCore x402 API" in landing.text

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True

    pricing = client.get("/pricing")
    assert pricing.status_code == 200
    assert pricing.json()["x402"]["price"] == DEFAULT_PRICE
    assert pricing.json()["token_offer"]["ca"] == LEOS_TOKEN_CA
    assert pricing.json()["token_offer"]["price"] == LEOS_TOKEN_PRICE
    assert pricing.json()["token_offer"]["enabled"] is True
    assert pricing.json()["tenancy"]["default_tenant"] == DEFAULT_TENANT_ID

    blocked = client.post("/leos/v1/route", json={"task": "search local agent memory"})
    assert blocked.status_code == 401
    assert client.post("/leos/v1/recall", json={"query": "memory"}).status_code == 401
    assert client.get("/leos/v1/dashboard").status_code == 401

    leos_route = client.post(
        "/leos/v1/route",
        headers={LEOS_ACCESS_HEADER: "leos-secret"},
        json={"task": "search local agent memory"},
    )
    assert leos_route.status_code == 200
    assert leos_route.json()["tenant"] == DEFAULT_TENANT_ID


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
    assert pricing["token_offer"]["enabled"] is False
    assert all(not row["route"].split(" ", 1)[1].startswith("/leos/") for row in pricing["routes"])
    assert client.get("/leos/v1/dashboard").status_code == 503


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
