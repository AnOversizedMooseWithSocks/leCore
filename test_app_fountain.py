"""The fountain erasure-robustness demo is wired into the app: provisioned for
the requested loss, it recovers a real blob exactly, and reports the recovery
curve with its information-floor cliff."""
import app


def _client():
    app.app.testing = True
    return app.app.test_client()


def test_fountain_endpoint_recovers_exactly_when_provisioned():
    c = _client()
    for loss in (20, 40):
        r = c.post("/api/fountain", data={"loss": loss}).get_json()
        assert r["exact_recovery"] is True            # provisioned above the floor
        assert r["survivor_ratio"] >= 1.2
        assert r["loss_pct"] == loss


def test_fountain_curve_has_the_cliff():
    c = _client()
    r = c.post("/api/fountain", data={"loss": 30}).get_json()
    curve = {x["overhead"]: x["success"] for x in r["curve"]}
    assert curve[1.0] == 0.0                          # cannot decode from < k droplets
    assert curve[1.5] >= curve[1.1]                   # reliability rises with overhead


def test_page_includes_fountain_panel():
    c = _client()
    html = c.get("/").get_data(as_text=True)
    assert "Erasure robustness" in html
