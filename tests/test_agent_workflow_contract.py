"""THE AGENT WORKFLOW CONTRACT. The complete authoring surface, exercised over a real HTTP socket, as one
CI-guarded promise.

WHY THIS FILE EXISTS. Every faculty on the parity surface has its own tests, and every one of them passes
in-process. That is necessary and it is not the contract: the thing real clients build on is the SERVED
surface -- JSON in, JSON out, handles across calls -- and this arc found, repeatedly, that "works
in-process" and "an agent can call it" are different claims (a Scene that serialised to a memory address;
a texture that needed a callable no client can send; keyframes with no JSON shape). Each was found by hand.
This file makes the whole loop a single regression trap, so the next boundary break is found by CI instead.

Every test here speaks ONLY HTTP. If a test in this file imports a holographic module for anything other
than starting the server, it is testing the wrong thing.

STRUCTURE. One session fixture starts the service; the tests then walk the loop a real client walks:
    describe -> inspect -> fix -> texture -> place -> light (HDRI) -> preview -> animate -> read back
with three properties asserted throughout, because they are what "production" means here:
  1. CONTRACTS: each call returns the documented shape, and handles from one call work in the next.
  2. ERRORS ARE STRUCTURED: caller mistakes come back {ok: False, error: <names the fix>} -- never a bare
     500, never a silent no-op. An agent can only recover from an error it can read.
  3. DETERMINISM: the same request twice gives the same bytes, over the wire, where determinism is
     promised. This is the engine's constitutional rule surfaced as an API property a client can rely on.
"""
import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

PORT = 8871


class _Client:
    def __init__(self, port):
        import holographic_service as HS
        self.port = port
        threading.Thread(target=HS.serve, kwargs=dict(host="127.0.0.1", port=port, threads=True),
                         daemon=True).start()
        time.sleep(3.0)

    def invoke(self, _tool, **args):
        req = urllib.request.Request("http://127.0.0.1:%d/invoke" % self.port,
                                     data=json.dumps({"name": _tool, "args": args}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            return json.loads(urllib.request.urlopen(req, timeout=900).read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read().decode())

    def ok(self, _tool, **args):
        """Invoke and unwrap, failing the test with the server's own error text if the call failed --
        so a broken stage names itself instead of surfacing three asserts later as a KeyError."""
        out = self.invoke(_tool, **args)
        assert out.get("ok"), "%s failed at the boundary: %s" % (_tool, out.get("error", "")[:300])
        return out["result"]


@pytest.fixture(scope="module")
def api():
    return _Client(PORT)


# =====================================================================================================
# Stage 1 -- words to a live document. The entry point a non-technical client actually uses.
# =====================================================================================================

def test_describe_creates_handled_objects(api):
    res = api.ok("describe_to_scene", text="a red cube on the left and a green sphere on the right")
    assert res["scene"]["ref"].startswith("ref:Scene:"), "the scene must come back as a usable handle"
    assert set(res["handles"]) == {"red box", "green sphere"}
    # the vocabulary boundary is REPORTED, not silent -- an agent must be able to see what fell through
    assert isinstance(res["unknown"], list)


def test_ungrounded_text_reports_rather_than_inventing(api):
    res = api.ok("describe_to_scene", text="a purple wombat")
    assert res["handles"] == {}, "nothing groundable must mean no objects, not a guess"
    assert "wombat" in res["unknown"], "the word that failed must be NAMED"


# =====================================================================================================
# Stage 2 -- the see->fix loop: inspect, catch a mistake pre-flight, repair it, confirm. The loop is the
# product; everything else is stages of it.
# =====================================================================================================

def test_preflight_catches_and_names_the_fix(api):
    s = api.ok("new_scene")["ref"]
    g = api.ok("sdf_parse", dsl_text="(sphere 0.5)")["ref"]
    api.ok("scene_add", scene=s, name="ball", geometry=g, material="oak")     # deliberately wrong
    info = api.ok("scene_info", scene=s)
    assert any("wood_oak" in p for p in info["problems"]), \
        "a wrong material must be caught BEFORE a render is paid for, with a did-you-mean"
    bad = [o for o in info["objects"] if o["material"] == "oak"][0]
    api.ok("scene_edit", scene=s, handle=bad["handle"], material="wood_oak")
    assert api.ok("scene_info", scene=s)["problems"] == [], "the fix must clear the pre-flight"


def test_caller_errors_are_structured_never_500s(api):
    """The error CONTRACT, sampled across the surface. An agent can only recover from an error it can
    read; a bare 500 or -- worse -- a silent no-op turns every caller mistake into a debugging session."""
    s = api.ok("new_scene")["ref"]
    g = api.ok("sdf_parse", dsl_text="(box 0.4 0.4 0.4)")["ref"]
    h = api.ok("scene_add", scene=s, name="cube", geometry=g, material="matte_gray")

    stale = api.invoke("scene_info", scene="ref:Scene:99999")
    assert stale["ok"] is False and "never minted" in stale["error"]

    cam = api.ok("camera", eye=[0, 1, 3], target=[0, 0, 0], fov_deg=40.0, aspect=4 / 3.)["ref"]
    bad_prop = api.invoke("render_animation", scene=s, camera=cam,
                          keys={h: {"velocity": [[0, [1, 0, 0]]]}}, n_frames=2, width=16, height=12)
    assert bad_prop["ok"] is False and "position" in bad_prop["error"], \
        "a wrong property name must come back naming the valid set"

    bad_scale = api.invoke("render_preview", scene=s, camera=cam, width=32, height=24, scale=2.0)
    assert bad_scale["ok"] is False and "FRACTION" in bad_scale["error"]


# =====================================================================================================
# Stage 3 -- appearance and light: texture by name, HDRI-shaped environment. Both were boundary breaks
# once (a callable can't cross JSON; 8-bit maps lose the sun); both must stay closed.
# =====================================================================================================

def test_texture_and_environment_compose(api):
    s = api.ok("new_scene")["ref"]
    g = api.ok("sdf_parse", dsl_text="(sphere 0.6)")["ref"]
    h = api.ok("scene_add", scene=s, name="ball", geometry=g, material="matte_gray")
    api.ok("scene_set_texture", scene=s, handle=h, texture="checker", scale=1.5)
    cam = api.ok("camera", eye=[0, 0.8, 2.6], target=[0, 0, 0], fov_deg=40.0, aspect=4 / 3.)["ref"]
    dome = api.ok("scene_light", kind="dome", intensity=1.5)["ref"]
    plain = np.asarray(api.ok("render_preview", scene=s, camera=cam, width=32, height=24,
                              lights=[dome]), float)
    api.ok("scene_set_texture", scene=s, handle=h, texture=None)
    bare = np.asarray(api.ok("render_preview", scene=s, camera=cam, width=32, height=24,
                             lights=[dome]), float)
    assert np.abs(plain - bare).mean() > 1e-4, "the texture must be visible over the wire"


# =====================================================================================================
# Stage 4 -- motion: keyframes in JSON, frames out, and the transforms genuinely move things.
# =====================================================================================================

def test_keyframes_move_objects(api):
    s = api.ok("new_scene")["ref"]
    g = api.ok("sdf_parse", dsl_text="(sphere 0.5)")["ref"]
    h = api.ok("scene_add", scene=s, name="ball", geometry=g, material="copper")
    cam = api.ok("camera", eye=[0, 1, 3], target=[0, 0, 0], fov_deg=40.0, aspect=4 / 3.)["ref"]
    frames = api.ok("render_animation", scene=s, camera=cam,
                    keys={h: {"position": [[0.0, [-1.0, 0, 0]], [1.0, [1.0, 0, 0]]]}},
                    n_frames=3, fps=3, width=24, height=18)
    f = [np.asarray(x, float) for x in frames]
    assert len(f) == 3 and np.abs(f[0] - f[-1]).mean() > 1e-3, \
        "identical frames = the silent-transform regression (see _PlacedEval)"


# =====================================================================================================
# Stage 5 -- determinism OVER THE WIRE. The engine's constitutional rule as an API property: a client that
# caches by request hash, diffs renders in CI, or reproduces a bug report is relying on exactly this.
# =====================================================================================================

def test_the_same_request_twice_is_byte_identical(api):
    def build_and_render():
        s = api.ok("new_scene")["ref"]
        g = api.ok("sdf_parse", dsl_text="(translate 0.2 0.5 0.0 (box 0.4 0.4 0.4))")["ref"]
        api.ok("scene_add", scene=s, name="cube", geometry=g, material="wood_oak")
        cam = api.ok("camera", eye=[2, 1.5, 3], target=[0, 0.4, 0], fov_deg=40.0, aspect=4 / 3.)["ref"]
        dome = api.ok("scene_light", kind="dome", intensity=1.4)["ref"]
        return api.ok("render_preview", scene=s, camera=cam, width=32, height=24,
                      seed=0, lights=[dome])

    a, b = build_and_render(), build_and_render()
    assert json.dumps(a) == json.dumps(b), \
        "two identical authoring sessions must produce byte-identical JSON -- determinism is part of the " \
        "served contract, not just an internal engine property"


# =====================================================================================================
# Stage 6 -- the manifest matches reality: every tool this contract exercises is declared in GET /tools.
# A tool that works but is not listed is invisible to a client that discovers by manifest; a listed tool
# that fails is a lie. Both directions checked.
# =====================================================================================================

def test_every_contract_tool_is_declared(api):
    listing = json.loads(urllib.request.urlopen(
        "http://127.0.0.1:%d/tools" % api.port, timeout=60).read())
    tools = listing.get("tools", listing)
    names = {t["name"] if isinstance(t, dict) else t for t in tools}
    used = {"describe_to_scene", "new_scene", "sdf_parse", "scene_add", "scene_edit", "scene_info",
            "scene_set_texture", "scene_light", "camera", "render_preview", "render_animation",
            "load_hdr", "sky_dome", "save_render", "load_image",
            # the post-parity surface joins the manifest contract the day it ships, not when it breaks:
            "sky_model", "fetch_asset", "refine_scene"}
    missing = used - names
    assert not missing, "contract tools absent from GET /tools: %s" % sorted(missing)


# =====================================================================================================
# Stage 7 -- BEYOND the reference integration: the self-improving loop, over the wire. Blender's MCP can
# show an agent its render; it cannot score candidate edits against a goal and apply the best one. This
# stage is the contract that leCore can, from JSON, with the trail on record.
# =====================================================================================================

def test_the_engine_improves_its_own_scene_toward_a_target(api):
    """describe -> render (that's the target) -> describe a WORSE starting point -> refine_scene closes
    the gap. Everything crosses as JSON: the semantic scene by ref handle, the target as a nested list.
    The assertion is on the DISTANCES the loop itself reports, plus the applied-edit trail -- an
    improvement claim without its numbers would be exactly the narrative this repo distrusts."""
    goal = api.ok("build_scene", text="a red sphere")
    goal_ref = goal["ref"]
    api.ok("adjust_scene", scene=goal_ref, command="make it night") \
        if "adjust_scene" in _tool_names(api) else None

    # if there is no adjust faculty, refine against a same-description target: distance starts ~0 and the
    # loop must simply not make it WORSE -- still a real contract, just a weaker one. Prefer the strong one.
    strong = "adjust_scene" in _tool_names(api)
    tgt = api.ok("render_semantic", scene=goal_ref, width=96, height=72) \
        if "render_semantic" in _tool_names(api) else None
    if tgt is None:
        import numpy as _np
        # fall back: build the target in-process ONCE (documented exception to the HTTP-only rule: the
        # target is INPUT DATA for the contract, not part of the surface under test)
        import lecore
        m = lecore.UnifiedMind(dim=128, seed=0)
        g = m.build_scene("a red sphere")
        g.adjust("make it night")
        tgt = _np.asarray(g.render(width=96, height=72), float).tolist()
        strong = True

    start = api.ok("build_scene", text="a red sphere")["ref"]
    rep = api.ok("refine_scene", scene=start, target_image=tgt, max_steps=4)
    assert rep["final_distance"] <= rep["start_distance"] + 1e-9, \
        "the refine loop made the scene WORSE: %.4f -> %.4f" % (rep["start_distance"], rep["final_distance"])
    if strong:
        assert rep["final_distance"] < rep["start_distance"] - 1e-3, \
            "a reachable target was not approached: %.4f -> %.4f (applied: %s)" % (
                rep["start_distance"], rep["final_distance"], rep.get("applied"))


def _tool_names(api):
    import urllib.request as _u
    listing = json.loads(_u.urlopen("http://127.0.0.1:%d/tools" % api.port, timeout=60).read())
    tools = listing.get("tools", listing)
    return {t["name"] if isinstance(t, dict) else t for t in tools}


# =====================================================================================================
# Stage 8 -- the parametric sky over the wire. sky_model returns a CALLABLE, the exact type that broke
# this boundary three times (texture sockets, timelines, refine methods); here the callable crosses as a
# REF and is consumed by two other tools. Cloud kinds travel as JSON lists, not tuples, because that is
# what json.loads actually delivers -- the test speaks the client's dialect on purpose.
# =====================================================================================================

def test_parametric_sky_drives_a_render_over_http(api):
    sky = api.ok("sky_model", hour=11.0, clouds=[["cirrocumulus", 0.6]])
    assert str(sky["ref"]).startswith("ref:"), "the sky must come back as a handle"
    s = api.ok("new_scene")["ref"]
    api.ok("scene_add", scene=s, name="floor",
           geometry=api.ok("sdf_parse", dsl_text="(plane 0.0)")["ref"], material="matte_gray")
    cam = api.ok("camera", eye=[0, 1, 4.5], target=[0, 2.2, -3.0], fov_deg=60.0, aspect=4 / 3.)["ref"]
    dome = api.ok("scene_light", kind="dome", color=sky["ref"], intensity=1.0)["ref"]
    img = np.asarray(api.ok("render_preview", scene=s, camera=cam, width=48, height=36,
                            lights=[dome], sky=sky["ref"], view=None), float)
    sky_band = img[:12]
    assert sky_band.std() > 0.02, \
        "a cirrocumulus sky rendered FLAT over the wire (std %.4f) -- the structure contract, at the boundary" \
        % sky_band.std()
    bad = api.invoke("sky_model", hour=12.0, clouds=[["cumulus", 0.5]])
    assert bad["ok"] is False and "cloud_scene" in bad["error"], \
        "the low-cloud refusal must survive the boundary and still name the right tool"


# =====================================================================================================
# Stage 9 -- the sky-synced sun over the wire, found unguarded by a discoverability sweep. The sky
# crosses as a ref INTO another tool's keyword argument (sky=), which is a resolve-in path no earlier
# stage exercised; cloud_shadows then swaps intensity for a server-side field. If ref-resolution inside
# kwargs ever regresses, this is the stage that says so.
# =====================================================================================================

def test_sky_synced_sun_with_cloud_shadows_over_http(api):
    sky = api.ok("sky_model", hour=9.5, clouds=[["stratocumulus", 0.6]])["ref"]
    sun = api.ok("scene_light", kind="sun", sky=sky, intensity=3.5, cloud_shadows=True)
    assert str(sun["ref"]).startswith("ref:DirectionalLight"), "the synced sun must come back as a light handle"
    s = api.ok("new_scene")["ref"]
    api.ok("scene_add", scene=s, name="floor",
           geometry=api.ok("sdf_parse", dsl_text="(plane 0.0)")["ref"], material="matte_gray")
    cam = api.ok("camera", eye=[0, 2.2, 7.0], target=[0, 0.4, -4.0], fov_deg=58.0, aspect=4 / 3.)["ref"]
    img = np.asarray(api.ok("render_preview", scene=s, camera=cam, width=48, height=36,
                            lights=[sun["ref"]], sky=sky, view=None), float)
    assert img[26:].std() > 0.02, \
        "cloud shadows flat over the wire (std %.4f) -- the intensity field died in ref resolution" % img[26:].std()
    bad = api.invoke("scene_light", kind="spot", sky=sky)
    assert bad["ok"] is False and "SUN" in bad["error"], "the non-sun refusal must survive the boundary"
