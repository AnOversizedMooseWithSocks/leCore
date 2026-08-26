"""The fractal-dimension vision demo is wired into the app: natural photos read
rougher (higher edge fractal dimension) than synthetic shapes."""
import app


def _client():
    app.app.testing = True
    return app.app.test_client()


def test_fractal_vision_demo_separates_natural_from_synthetic():
    c = _client()
    r = c.post("/api/vision", data={"demo": "fractal"}).get_json()
    assert r["demo"] == "fractal"
    assert r["rows"] and any(x["kind"] == "synthetic" for x in r["rows"])
    if r["natural_mean"] is not None:                 # photo samples present
        assert r["natural_mean"] > r["synthetic_mean"]
        assert r["natural_mean"] > 1.3


def test_page_includes_fractal_button():
    c = _client()
    html = c.get("/").get_data(as_text=True)
    assert 'data-d="fractal"' in html
