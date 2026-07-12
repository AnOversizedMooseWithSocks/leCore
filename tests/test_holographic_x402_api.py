"""Tests for the optional x402-paid API publisher."""

import pytest

from holographic_x402_api import (
    DEFAULT_NETWORK,
    DEFAULT_PRICE,
    LEOS_SITE_URL,
    LEOS_TOKEN_CA,
    LEOS_TOKEN_PRICE,
    X402Config,
    create_app,
    landing_page_html,
    leos_token_offer,
    optional_dependency_help,
    payment_manifest,
)


def test_default_x402_config_uses_testnet_price_shape():
    cfg = X402Config(pay_to="0xabc")

    assert cfg.network == DEFAULT_NETWORK
    assert cfg.price == DEFAULT_PRICE and cfg.price.startswith("$")
    assert cfg.facilitator_url == "https://x402.org/facilitator"


def test_payment_manifest_protects_specific_read_routes_only():
    manifest = payment_manifest(X402Config(pay_to="0xabc"))
    routes = {row["route"] for row in manifest}

    assert routes == {"POST /v1/recall", "POST /v1/route", "GET /v1/dashboard"}
    assert all("*" not in route for route in routes)
    assert "POST /admin/remember" not in routes
    assert "GET /health" not in routes
    assert all(row["accepts"][0]["pay_to"] == "0xabc" for row in manifest)


def test_price_validation_keeps_x402_format_honest():
    with pytest.raises(ValueError, match="dollar prefix"):
        X402Config(pay_to="0xabc", price="0.001")


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
