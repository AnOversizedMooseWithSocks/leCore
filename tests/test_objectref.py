"""Regression traps for object handles over /invoke (J-3D-24).

The claim being defended is narrow and testable: what /invoke hands back can be posted straight into the
next /invoke, for objects JSON cannot carry. Before this, POST /invoke new_scene returned a memory address
and the entire Scene-document family was listed in GET /tools and impossible to call.
"""
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from holographic.io_and_interop.holographic_objectref import ObjectRefs, is_ref


def test_a_handle_returns_the_same_object_not_a_copy():
    """Identity, not equality. A registry that returned a copy would break every mutation: an agent would
    add to one Scene and render another, with nothing anywhere to say so."""
    r = ObjectRefs()
    obj = {"live": True}
    h = r.put(obj)
    assert r.get(h) is obj
    obj["live"] = False
    assert r.get(h)["live"] is False, "the handle must track the LIVE object"


def test_handles_are_never_reused():
    """THE dangerous failure this design exists to prevent. With id() as the handle, a freed object's
    address is recycled and a stale handle silently resolves to a DIFFERENT object -- a wrong answer that
    looks completely right. A monotonic counter cannot do that, even under eviction."""
    r = ObjectRefs(capacity=2)
    seen = set()
    for _ in range(10):
        seen.add(r.put(object()))
    assert len(seen) == 10, "every handle must be unique across the whole process lifetime"
    assert r.stats()["live"] == 2 and r.stats()["minted"] == 10


def test_eviction_is_distinguishable_from_never_existed():
    """Two failures, two different fixes -- raise the capacity, or re-create the object. An error that
    blurs them sends an agent down the wrong path, which is worse than a slightly terser message."""
    r = ObjectRefs(capacity=1)
    old = r.put(object())
    r.put(object())
    with pytest.raises(KeyError, match="EVICTED"):
        r.get(old)
    with pytest.raises(KeyError, match="never minted"):
        r.get("ref:Scene:9999")


def test_plain_strings_are_left_alone():
    """Silently reinterpreting a caller's text as a handle is a wrong answer, not a bug. Only the 'ref:'
    prefix plus a KNOWN handle resolves; everything else passes through untouched."""
    r = ObjectRefs()
    h = r.put([1, 2, 3])
    out = r.resolve({"a": h, "b": "reference", "c": "/tmp/ref.png", "d": ["plain", 7]})
    assert out["a"] == [1, 2, 3]
    assert out["b"] == "reference" and out["c"] == "/tmp/ref.png" and out["d"] == ["plain", 7]
    assert is_ref(h) and not is_ref("reference")


def test_a_typo_raises_rather_than_passing_through():
    """A faculty receiving the literal text 'ref:Scene:9' fails somewhere confusing, minutes from the
    actual mistake. Fail at the boundary, where the message can still name the cause."""
    with pytest.raises(KeyError):
        ObjectRefs().resolve({"scene": "ref:Scene:9"})


def test_jsonable_without_a_registry_is_unchanged():
    """ADDITIVITY at the HTTP boundary. `refs=None` must reproduce the exact dict shipped before this
    existed -- an existing client that reads 'type' and 'repr' sees precisely the keys it always saw."""
    import holographic_service as HS

    class Opaque:
        def __repr__(self):
            return "<Opaque>"

    assert HS._jsonable(Opaque()) == {"type": "Opaque", "repr": "<Opaque>"}
    with_ref = HS._jsonable(Opaque(), ObjectRefs())
    assert with_ref["type"] == "Opaque" and with_ref["repr"] == "<Opaque>"
    assert with_ref["ref"].startswith("ref:Opaque:"), "the handle is ADDED, never a replacement"


class _Server:
    """A real service on a real port. The point of this file is the HTTP boundary, and an in-process call
    would test everything except the thing that was broken."""

    def __init__(self, port):
        import holographic_service as HS
        self.port = port
        threading.Thread(target=HS.serve,
                         kwargs=dict(host="127.0.0.1", port=port, threads=True), daemon=True).start()
        time.sleep(3.0)

    def invoke(self, _tool, **args):
        req = urllib.request.Request("http://127.0.0.1:%d/invoke" % self.port,
                                     data=json.dumps({"name": _tool, "args": args}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            return json.loads(urllib.request.urlopen(req, timeout=600).read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read().decode())


@pytest.fixture(scope="module")
def server():
    return _Server(8779)


def test_an_http_only_agent_can_author_and_fix_a_scene(server):
    """THE END-TO-END CLAIM, and the reason this item existed. Nothing here imports lecore: an agent with
    only POST /invoke mints a Scene, builds geometry, adds it, READS the scene back, acts on the pre-flight
    warning, and confirms the fix. Every previous step of this arc was in-process-only without it."""
    scene = server.invoke("new_scene")["result"]["ref"]
    assert scene.startswith("ref:Scene:")

    ball = server.invoke("sdf_parse", dsl_text="(sphere 0.6)")["result"]["ref"]
    cube = server.invoke("sdf_parse", dsl_text="(box 0.4 0.4 0.4)")["result"]["ref"]
    assert server.invoke("scene_add", scene=scene, name="ball", geometry=ball,
                         material="copper")["ok"]
    # 'oak' is deliberately wrong -- the real library name is 'wood_oak'
    assert server.invoke("scene_add", scene=scene, name="cube", geometry=cube, material="oak")["ok"]

    info = server.invoke("scene_info", scene=scene)["result"]
    assert info["n_objects"] == 2 and info["empty"] is False
    assert any("wood_oak" in p for p in info["problems"]), "the pre-flight must catch it over HTTP too"

    bad = [o for o in info["objects"] if o["material"] == "oak"][0]
    assert server.invoke("scene_edit", scene=scene, handle=bad["handle"], material="wood_oak")["ok"]
    assert server.invoke("scene_info", scene=scene)["result"]["problems"] == [], \
        "the agent fixed its own mistake without a human and without paying for a render"


def test_a_bad_handle_is_a_caller_error_not_a_500(server):
    """An agent that gets an opaque 500 retries blindly. {ok: False, error} with a message naming the cause
    is the difference between a recoverable mistake and a dead end."""
    out = server.invoke("scene_info", scene="ref:Scene:999999")
    assert out["ok"] is False and "never minted" in out["error"]
