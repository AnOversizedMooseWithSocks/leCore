"""Tests for the optional x402-paid API publisher."""

import pytest

from holographic_x402_api import (
    DEFAULT_NETWORK,
    DEFAULT_PRICE,
    DEFAULT_TENANT_ID,
    LEOS_SITE_URL,
    LEOS_TOKEN_CA,
    LEOS_TOKEN_PRICE,
    TENANT_HEADER,
    TENANT_TOKEN_HEADER,
    X402Config,
    create_app,
    landing_page_html,
    leos_token_offer,
    optional_dependency_help,
    payment_manifest,
    tenant_access_token,
    x402_route_configs,
)


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


def test_leos_token_offer_is_ca_only_metadata():
    offer = leos_token_offer()

    assert offer["site"] == LEOS_SITE_URL
    assert offer["ca"] == LEOS_TOKEN_CA
    assert offer["price"] == LEOS_TOKEN_PRICE
    assert "POST /leos/v1/recall" in offer["discount_routes"]
    assert "Only the CA is needed" in offer["note"]


def test_unpaid_dev_app_serves_landing_page_and_keeps_api_routes_free():
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    client = fastapi_testclient.TestClient(
        create_app(config=X402Config(pay_to="0xabc"), paid=False)
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
    assert pricing.json()["tenancy"]["default_tenant"] == DEFAULT_TENANT_ID

    leos_route = client.post("/leos/v1/route", json={"task": "search local agent memory"})
    assert leos_route.status_code == 200
    assert leos_route.json()["tenant"] == DEFAULT_TENANT_ID


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
