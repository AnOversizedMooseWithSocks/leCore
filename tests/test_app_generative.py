"""The generative work is wired into the live app, not just the library: compose, morph,
and nucleus endpoints respond and return what their panels render. These tests hit the
actual Flask routes (the wiring), so a broken endpoint fails the build. compose and morph
need no loaded mind; nucleus reports it needs a dataset until one is loaded.
"""
import base64

import tools.unified_app as ua


def _client():
    ua.app.testing = True
    return ua.app.test_client()


def _is_png_data_url(s):
    return isinstance(s, str) and s.startswith("data:image/png;base64,") and \
        len(base64.b64decode(s.split(",", 1)[1])) > 100


def test_compose_panel_is_present_and_wired():
    c = _client()
    page = c.get("/").data
    assert b"compose a scene" in page
    assert b"async function composeScene" in page          # JS handler embedded
    assert b"/api/unified/compose" in page                 # button posts to the route


def test_compose_endpoint_round_trips_a_novel_scene():
    c = _client()
    r = c.post("/api/unified/compose", json={"objects": 3, "seed": 1}).get_json()
    assert r["exact"] is True                               # composed scene factors back exactly
    assert len(r["composed"]) == 3 and len(r["recovered"]) == 3
    assert _is_png_data_url(r["image"])                     # a real rendered image came back
    assert r["render_shape_ok"] and r["render_colour_ok"]  # pixels auto-tag to the spec
    assert r["anim_faithful"] == 1.0                        # the colour sweep is on-target
    assert all(_is_png_data_url(u) for u in r["anim_frames"])


def test_morph_panel_is_present_and_wired():
    c = _client()
    page = c.get("/").data
    assert b"slerp in the coefficient domain" in page
    assert b"async function morphScene" in page
    assert b"/api/unified/morph" in page


def test_morph_endpoint_beats_crossfade_on_ghosting():
    c = _client()
    r = c.post("/api/unified/morph", json={"seed": 2}).get_json()
    assert r["ghost_crossfade"] < 1e-6                      # crossfade midpoint IS the ghost
    assert r["ghost_morph"] > r["ghost_crossfade"]         # coeff morph blends structure
    assert _is_png_data_url(r["morph_mid"]) and _is_png_data_url(r["crossfade_mid"])
    assert len(r["morph_frames"]) >= 5


def test_nucleus_panel_is_present_and_wired():
    c = _client()
    page = c.get("/").data
    assert b"nucleus text" in page
    assert b"async function nucleusGen" in page
    assert b"/api/unified/nucleus" in page


def test_nucleus_endpoint_asks_for_a_dataset_before_one_is_loaded():
    # Without a loaded mind the endpoint must say so, not crash.
    c = _client()
    ua.STATE["mind"] = None
    r = c.post("/api/unified/nucleus", json={"seed": "the "}).get_json()
    assert "error" in r and "dataset" in r["error"].lower()


def test_persist_panel_is_present_and_wired():
    c = _client()
    page = c.get("/").data
    assert b"save &amp; reload" in page
    assert b"async function persistMind" in page
    assert b"/api/unified/persist" in page


def test_persist_endpoint_round_trips_the_learned_meaning_space():
    # Load a small mind that learns, then save+reload the WHOLE memory through the
    # versioned core and confirm it survives identically -- same classifications and word
    # neighbours. This is holographic_core.save/load wired into the app over the real
    # SelfOrganizingMind, not just a word-vector slice.
    c = _client()
    loaded = c.post("/api/unified/load", json={"id": "curriculum"}).get_json()
    assert loaded.get("ok") is True
    r = c.post("/api/unified/persist", json={}).get_json()
    assert "error" not in r, r
    assert r["prototypes"] > 0                          # the learned prototype bank persisted
    assert r["same_classifications"] is True            # ...and classifies identically on reload
    assert r["same_neighbours"] is True                 # word neighbourhoods survive too
    assert r["version_guard_works"] is True             # a bad format version is refused


def test_persist_endpoint_asks_for_a_dataset_first():
    c = _client()
    ua.STATE["mind"] = None
    r = c.post("/api/unified/persist", json={}).get_json()
    assert "error" in r and "dataset" in r["error"].lower()


def test_nested_panel_is_present_and_wired():
    c = _client()
    page = c.get("/").data
    assert b"nested scene" in page
    assert b"async function nestedScene" in page
    assert b"/api/unified/nested" in page


def test_nested_endpoint_round_trips_scene_of_scenes():
    # Fractal composition wired into the app: compose a scene-of-scenes and factor each
    # group back. 2 groups recover exactly (the measured near-perfect regime).
    c = _client()
    r = c.post("/api/unified/nested", json={"groups": 2, "per_group": 2, "seed": 5}).get_json()
    assert "error" not in r, r
    assert r["total_groups"] == 2
    assert r["exact_groups"] == 2                         # both sub-scenes recovered exactly
    assert all(_is_png_data_url(g["image"]) for g in r["groups"])
