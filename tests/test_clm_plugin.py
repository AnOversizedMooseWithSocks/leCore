"""E6.2 / E6.3 -- the `clm` plugin (holographic/plugins/clm.py), CONTRACT-TESTED WITH A STUB.

No GPU, no torch, no real CLM here -- and no live numbers are claimed anywhere in this file. An in-process stub
HTTP server (http.server on 127.0.0.1, a random port, a daemon thread) speaks the ASSUMED CLM dialect the plugin's
docstring spells out (clm-serve's /v1/systemone, /v1/rank, /health, /v1/models, plus the encoder's OpenAI-style
/v1/embeddings). The tests pin what the plugin promises regardless of which CLM sits behind it:

  * a valid reply becomes leCore's typed record (value, ranked, margin, p_correct from OUR gate, id, evidence);
  * every malformed reply is REFUSED with a reason, never repaired, never raised out of the verb;
  * a slow server hits the timeout and a dead one is 'unreachable' -- both refusals, both bounded in time;
  * without torch/transformers/contrastive-lm the plugin still loads, plugin_list says available=False, the local
    backend reports unavailable cleanly, and NOTHING optional gets imported;
  * the reply is learnable: decision_outcome(id) calibrates the CLM door; as an escalate= tier inside m.typed a
    dead or off-schema CLM becomes via='refused', not an exception;
  * E6.3: set_embedder's own gate decides -- a deterministic hash projection FAILS it (chance-level self-recall),
    a stub serving the index's own rows PASSES it (so the transport is not what fails).
Fake secrets only: "Hunter2-FAKE-9c1d".
"""
import hashlib
import json
import math
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

import lecore
from holographic.plugins import clm as clm_mod

FAKE_KEY = "Hunter2-FAKE-9c1d"


# ---------------------------------------------------------------------------------------------------------------
# the stub CLM server
# ---------------------------------------------------------------------------------------------------------------
def _words(t):
    return set(re.findall(r"[a-z]+", str(t).lower()))


def _softmax(xs, tau=0.5):
    m = max(xs)
    e = [math.exp((x - m) / tau) for x in xs]
    s = sum(e)
    return [v / s for v in e]


class Stub:
    """Deterministic stand-in: scores options by word overlap between the state and each option's criteria text,
    softmax -> probabilities. `mode` switches in one malformation at a time."""

    def __init__(self):
        self.mode = "ok"
        self.delay = 0.0
        self.emb = "hash128"
        self.oracle = {}
        self.seen = []
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj, raw=None):
                body = raw if raw is not None else json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json" if raw is None else "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-CLM-Latency-Ms", "1.5")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                stub.seen.append((self.path, self.headers.get("Authorization"), None))
                if self.path == "/health":
                    return self._send(200, {"ok": True, "embedder": True, "models": ["clm-latest"], "cache": None})
                if self.path == "/v1/models":
                    return self._send(200, {"models": [{"name": "clm-latest"}]})
                return self._send(404, {"error": "no route"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n) or b"{}")
                stub.seen.append((self.path, self.headers.get("Authorization"), body))
                if stub.delay:
                    time.sleep(stub.delay)
                if stub.mode == "http500":
                    return self._send(500, {"error": "boom"})
                if stub.mode == "html":
                    return self._send(200, None, raw=b"<html>not json</html>")
                if self.path == "/v1/systemone":
                    return self._send(200, stub.systemone(body))
                if self.path == "/v1/rank":
                    return self._send(200, stub.rank(body))
                if self.path == "/v1/embeddings":
                    return self._send(200, stub.embeddings(body))
                return self._send(404, {"error": "no route"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    # ---- the assumed CLM dialect -----------------------------------------------------------------------------
    def systemone(self, body):
        st = _words(body["state"])
        answers = {}
        for qid, q in body["questions"].items():
            if self.mode == "missing":
                continue
            if q["type"] == "choice":
                opts = list(q["criteria"])
                pr = _softmax([len(st & _words(o + " " + str(q["criteria"][o]))) for o in opts])
                probs = dict(zip(opts, pr))
                choice = opts[int(np.argmax(pr))]
                if self.mode == "bad_choice":
                    choice = "not_an_option"
                elif self.mode == "bad_keys":
                    probs.pop(opts[-1])
                elif self.mode == "bad_sum":
                    probs = {k: 2 * v for k, v in probs.items()}
                elif self.mode == "not_argmax":
                    choice = opts[int(np.argmin(pr))]
                answers[qid] = {"choice": choice, "confidence": max(pr), "probabilities": probs}
            elif q["type"] == "noul":
                crit = q.get("criteria") or {}
                p = _softmax([len(st & _words(crit.get("true", "yes"))), len(st & _words(crit.get("false", "no")))])[0]
                answers[qid] = {"noul": 1.7 if self.mode == "bad_noul" else p}
            else:
                levels = q["criteria"]
                pr = _softmax([len(st & _words(l)) for l in levels])
                e = sum(i * p for i, p in enumerate(pr))
                answers[qid] = {"score": len(levels) + 3.0 if self.mode == "bad_score" else e,
                                "confidence": max(pr), "legend": dict(enumerate(levels)), "probabilities": pr}
        return {"model": body.get("model"), "answers": answers, "usage": {"input_tokens": 12}}

    def rank(self, body):
        st = _words(body["context"])
        cands = list(body["answers"])
        pr = _softmax([len(st & _words(c)) for c in cands])
        order = sorted(range(len(cands)), key=lambda i: -pr[i])
        rows = [{"rank": j + 1, "candidate": cands[i], "prob": pr[i]} for j, i in enumerate(order)]
        if self.mode == "rank_dup":
            rows[1]["candidate"] = rows[0]["candidate"]
        elif self.mode == "rank_invent":
            rows[-1]["candidate"] = "ghost action"
        elif self.mode == "rank_drop":
            rows = rows[:-1]
        elif self.mode == "rank_unordered":
            rows = [dict(r, rank=j + 1) for j, r in enumerate(rows[::-1])]    # renumbered, so ORDER is the defect
        return {"model": body.get("model"), "ranked": rows}

    def embeddings(self, body):
        data = []
        for t in body["input"]:
            if self.emb == "oracle":
                v = self.oracle.get(" ".join(str(t).split()), np.zeros(128))
            elif self.emb in ("hash128", "hash4096"):
                # A DETERMINISTIC HASH PROJECTION: the bag of words through sha256-seeded Gaussian directions.
                # Same text -> same vector, similar texts -> similar vectors -- and no relation to nomic's space.
                d = 128 if self.emb == "hash128" else 4096
                v = np.zeros(d)
                for w in sorted(_words(t)):
                    seed = int.from_bytes(hashlib.sha256(w.encode()).digest()[:8], "big") % (2 ** 32)
                    v += np.random.default_rng(seed).standard_normal(d)
            elif self.emb == "nan":
                v = np.full(8, np.nan)
            else:
                v = np.zeros(0)
            data.append({"object": "embedding", "index": len(data), "embedding": [float(x) for x in v]})
        return {"object": "list", "data": data, "model": body.get("model")}

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture(scope="module")
def stub():
    s = Stub()
    yield s
    s.close()


@pytest.fixture
def fresh(stub):
    stub.mode, stub.delay, stub.emb, stub.oracle = "ok", 0.0, "hash128", {}
    stub.seen.clear()
    return stub


def _mind(_catalog=False, **cfg):
    """A slim mind with only this plugin. Catalog registration is skipped except where discoverability is the
    subject: it costs ~1.3-1.6 s per load on a loaded box (the plugin itself loads in ~1 ms)."""
    m = lecore.UnifiedMind(dim=64, seed=0, plugins=())
    m._plugin_load("clm", config=cfg, register_in_catalog=_catalog)
    return m


CHOICE = {"topic": {"type": "choice", "options": ["billing", "shipping"],
                    "examples": {"billing": ["charged twice on my card"], "shipping": ["courier lost my parcel"]}}}


# ---------------------------------------------------------------------------------------------------------------
# loading, preflight, no optional imports
# ---------------------------------------------------------------------------------------------------------------
def test_plugin_loads_without_its_requirements_and_imports_nothing_optional():
    before = set(sys.modules)
    m = _mind(backend="local")
    rec = [p for p in m.plugin_list() if p["name"] == "clm"][0]
    missing = clm_mod._missing_local()
    if missing:
        assert rec["available"] is False and set(missing) <= set(rec["missing"]) | {"clm"}
        assert "pip install" in rec["install"]
    for v in ("clm_status", "clm_systemone", "clm_rank", "clm_escalator", "clm_embedder"):
        assert callable(getattr(m, v)), v
    st = m.clm_status()                                        # never raises
    if missing:
        assert st["reachable"] is False and "not installed" in st["why"]
        assert st["local_missing"] == missing
    assert not ({"torch", "transformers", "clm"} & (set(sys.modules) - before)), "an optional package was imported"


def test_the_default_mind_discovers_the_plugin():
    names = [p["name"] for p in lecore.UnifiedMind(dim=64, seed=0).plugin_list()]
    assert "clm" in names


def test_local_backend_refuses_cleanly_when_packages_are_missing():
    if not clm_mod._missing_local():
        pytest.skip("torch/transformers/contrastive-lm are installed here; this pins the ABSENT case")
    before = set(sys.modules)
    m = _mind(backend="local")
    out = m.clm_systemone("courier lost my parcel", CHOICE)
    assert out["topic"]["via"] == "refused" and out["topic"]["refusal"] == "unavailable"
    assert "not installed" in out["topic"]["why"]
    assert m.clm_rank("x", ["a", "b"])["via"] == "refused"
    with pytest.raises(clm_mod.CLMUnavailable):
        m.clm_embedder()
    with pytest.raises(clm_mod.CLMUnavailable):
        _mind(url="http://127.0.0.1:9").clm_embedder(space="state")     # the state head always runs locally
    assert not ({"torch", "transformers", "clm"} & (set(sys.modules) - before))


def test_bad_config_is_refused_at_load_but_a_bad_env_never_breaks_a_mind(monkeypatch):
    with pytest.raises(Exception):
        _mind(backend="gpu-please")                           # the operator asked explicitly: refuse loudly
    with pytest.raises(Exception):
        _mind(timeout="soon")
    # every default mind discovers this plugin, and discovery has no try/except -- a typo in the environment must
    # degrade to the default with a warning, never make UnifiedMind() unconstructable
    monkeypatch.setenv("LECORE_CLM_TIMEOUT", "abc")
    monkeypatch.setenv("LECORE_CLM_BACKEND", "gpu")
    m = lecore.UnifiedMind(dim=64, seed=0)
    st = m.clm_status()
    assert st["timeout"] == 10.0 and st["backend"] in ("http", "local")
    assert len(st["config_warnings"]) == 2 and "LECORE_CLM_TIMEOUT" in st["config_warnings"][0]


def test_status_over_http(fresh):
    m = _mind(url=fresh.url, api_key=FAKE_KEY)
    st = m.clm_status()
    assert st["backend"] == "http" and st["reachable"] is True and st["health"]["ok"] is True
    assert st["auth"] is True
    assert FAKE_KEY not in json.dumps(st), "the API key leaked into status"
    assert fresh.seen[-1][1] == "Bearer " + FAKE_KEY, "the key must reach the server as a Bearer header"
    assert _mind(backend="http").clm_status()["why"].startswith("http backend has no URL")


# ---------------------------------------------------------------------------------------------------------------
# E6.2 -- typed records, validation, refusal
# ---------------------------------------------------------------------------------------------------------------
def test_choice_becomes_a_typed_record(fresh):
    m = _mind(url=fresh.url)
    r = m.clm_systemone("the courier lost my parcel", CHOICE)["topic"]
    assert r["value"] == "shipping" and r["via"] == "model_end" and r["tier"] == "clm"
    assert [n for n, _ in r["ranked"]] == ["shipping", "billing"]
    assert r["margin"] == pytest.approx(r["ranked"][0][1] - r["ranked"][1][1])
    assert r["p_correct"] is None and r["calibrated"] is False       # never CLM's own number
    assert r["evidence"]["model_confidence"] == pytest.approx(r["ranked"][0][1])
    assert m.decision_ledger().get(r["id"]).via == "model_end"
    sent = fresh.seen[-1][2]["questions"]["topic"]
    assert sent["type"] == "choice" and sent["criteria"]["shipping"] == "courier lost my parcel"


def test_noul_and_score_map_back_to_our_types(fresh):
    m = _mind(url=fresh.url)
    q = {"urgent": {"type": "noul", "examples": {"yes": ["asap emergency now"], "no": ["whenever convenient"]}},
         "severity": {"type": "score", "min": 0, "max": 10,
                      "anchors": [["minor cosmetic issue", 1], ["service is down for everyone", 9]]}}
    out = m.clm_systemone("emergency: the service is down for everyone now", q)
    assert out["urgent"]["value"] is True and out["urgent"]["ranked"][0][0] == "yes"
    s = out["severity"]
    assert s["type"] == "score" and 1.0 <= s["value"] <= 9.0 and s["value"] > 5.0
    assert s["ranked"][0][0] == "service is down for everyone"


@pytest.mark.parametrize("mode,needle", [
    ("bad_choice", "not one of"),
    ("bad_keys", "keyed by exactly the options"),
    ("bad_sum", "not a probability"),
    ("not_argmax", "inconsistent"),
    ("missing", "no answer for"),
    ("html", "not JSON"),
    ("http500", "HTTP 500"),
])
def test_malformed_replies_are_refused_with_a_reason(fresh, mode, needle):
    fresh.mode = mode
    r = _mind(url=fresh.url).clm_systemone("the courier lost my parcel", CHOICE)["topic"]
    assert r["via"] == "refused" and r["value"] is None and r["id"] is None
    assert needle in r["why"], r["why"]


@pytest.mark.parametrize("mode,q", [
    ("bad_noul", {"u": {"type": "noul"}}),
    ("bad_score", {"s": {"type": "score", "min": 0, "max": 1}}),
])
def test_out_of_range_numbers_are_refused(fresh, mode, q):
    fresh.mode = mode
    (r,) = _mind(url=fresh.url).clm_systemone("anything at all", q).values()
    assert r["via"] == "refused"


def test_our_schema_is_checked_before_asking(fresh):
    with pytest.raises(Exception):
        _mind(url=fresh.url).clm_systemone("x", {"q": {"type": "choice", "options": ["only-one"]}})
    assert not fresh.seen, "an invalid question must never reach the model"


def test_timeout_and_unreachable_are_bounded_refusals(fresh):
    fresh.delay = 1.5
    m = _mind(url=fresh.url, timeout=0.3)
    t0 = time.time()
    r = m.clm_systemone("the courier lost my parcel", CHOICE)["topic"]
    assert r["via"] == "refused" and "timed out" in r["why"] and r["refusal"] == "unavailable"
    assert time.time() - t0 < 1.4, "the timeout did not bound the call"
    fresh.delay = 0.0
    with socket_closed_port() as url:
        r = _mind(url=url, timeout=1.0).clm_systemone("x", CHOICE)["topic"]
        assert r["via"] == "refused" and "unreachable" in r["why"]
        assert _mind(url=url, timeout=1.0).clm_status()["reachable"] is False


class socket_closed_port:
    """A URL on a loopback port nobody listens on (bound then closed)."""

    def __enter__(self):
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        return "http://127.0.0.1:%d" % self.port

    def __exit__(self, *a):
        return False


def test_rank_is_validated_against_the_request(fresh):
    m = _mind(url=fresh.url)
    cands = ["revert the last commit", "open the settings page", "run the test suite"]
    r = m.clm_rank("please undo the last commit", cands, instructions="which action")
    assert r["value"] == "revert the last commit" and sorted(n for n, _ in r["ranked"]) == sorted(cands)
    assert r["via"] == "model_end" and r["id"] and fresh.seen[-1][2]["question"] == "which action"
    for mode, needle in (("rank_dup", "twice"), ("rank_invent", "not a candidate"), ("rank_drop", "rows for"),
                         ("rank_unordered", "best first")):
        fresh.mode = mode
        bad = m.clm_rank("please undo the last commit", cands)
        assert bad["via"] == "refused" and needle in bad["why"], (mode, bad["why"])
    with pytest.raises(ValueError):
        m.clm_rank("x", ["same", "same"])


# ---------------------------------------------------------------------------------------------------------------
# the reply is learnable
# ---------------------------------------------------------------------------------------------------------------
def test_reported_outcomes_calibrate_the_clm_door(fresh):
    m = _mind(url=fresh.url)
    # overlaps with the criteria differ from state to state, so CLM's margins differ -- an isotonic calibrator
    # refuses a window in which every score is identical (SystemOne keeps its previous fit in that case too)
    states = ["courier lost my parcel", "lost my parcel", "courier parcel", "parcel", "lost courier", "my parcel",
              "charged twice on my card", "charged my card", "twice charged", "card", "charged", "my card twice"]
    for i, s in enumerate(states):
        r = m.clm_systemone(s, CHOICE)["topic"]
        truth = r["value"] if i % 3 else ("billing" if r["value"] == "shipping" else "shipping")   # 1 in 3 wrong
        rep = m.decision_outcome(r["id"], truth)
        assert rep["forwarded"]["door"] == "clm:topic"
    cal = m.door_calibrator("clm:topic")
    assert cal.calibrated() and len(cal.pairs) == 12
    r = m.clm_systemone("my parcel never arrived from the courier", CHOICE)["topic"]
    assert r["p_correct"] is not None and 0.0 <= r["p_correct"] <= 1.0 and r["calibrated"] is True


def test_clm_as_the_escalation_tier_of_typed(fresh):
    m = _mind(url=fresh.url)
    ex = {"billing": ["charged twice on my card"], "shipping": ["courier lost my parcel"]}
    # decide_or_escalate strips examples from the payload, so without `examples=` CLM would score the bare option
    # NAMES (the stub then ties 0.5/0.5 and insertion order picks 'billing' -- a real CLM loses information too)
    bare = m.clm_escalator()
    t = m.typed("the courier lost my parcel", ["billing", "shipping"], margin=0.99, escalate=bare)
    assert t["via"] == "escalated" and bare.last["margin"] == pytest.approx(0.0)
    assert fresh.seen[-1][2]["questions"]["answer"]["criteria"] == {"billing": "billing", "shipping": "shipping"}
    esc = m.clm_escalator(examples=ex)
    # margin=0.99 makes the substrate abstain, so the question goes to the model end
    a = m.typed("the courier lost my parcel", ["billing", "shipping"], examples=ex, margin=0.99, escalate=esc)
    assert a["via"] == "escalated" and a["value"] == "shipping" and a.get("p_correct") is None
    assert esc.calls == 1 and esc.last["value"] == "shipping" and esc.last["id"] != a["id"]
    m.decision_outcome(a["id"], "shipping")                  # trains SystemOne + the reflex
    rep = m.decision_outcome(esc.last["id"], "shipping")     # calibrates the CLM tier
    assert rep["forwarded"]["door"] == "clm:answer"
    # a CLM that answers off-schema is REFUSED inside typed -- never an exception out of it
    fresh.mode = "bad_choice"
    b = m.typed("my card was charged twice", ["billing", "shipping"], margin=0.99, escalate=m.clm_escalator())
    assert b["via"] == "refused" and "not one of" in b["why"]
    fresh.mode = "ok"
    with socket_closed_port() as url:
        mm = _mind(url=url, timeout=1.0)
        c = mm.typed("x y z", ["billing", "shipping"], margin=0.99, escalate=mm.clm_escalator())
        assert c["via"] == "refused" and "unreachable" in c["why"]


# ---------------------------------------------------------------------------------------------------------------
# E6.3 -- CLM's encoder behind set_embedder: the gate decides
# ---------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def routing():
    """The shipped routing index and the module summaries its probe embeds (computed ONCE per module: the tree
    walk is ~5 s and is test_embedder_seam's subject, not this file's)."""
    m = lecore.UnifiedMind(dim=64, seed=0, plugins=())
    router = m._embedding_router()
    if router is None:
        pytest.skip("no routing index shipped")
    from holographic.semantic_router.holographic_embedseam import module_summaries
    names = [str(n) for n in router.names]
    summaries = module_summaries(".", names=set(names))
    data = np.load("lecore_data/routing/index_128d.npz", allow_pickle=True)
    q = data["q"].astype(np.float64)
    lo, hi = data["lo"].astype(np.float64), data["hi"].astype(np.float64)
    rows = lo + (hi - lo) * (q / 255.0)
    oracle = {summaries[n]: rows[i] for i, n in enumerate(names) if n in summaries}
    return {"summaries": summaries, "oracle": oracle}


@pytest.fixture
def fast_probe(routing, monkeypatch):
    import holographic.semantic_router.holographic_embedseam as es
    monkeypatch.setattr(es, "module_summaries",
                        lambda root, names=None: {k: v for k, v in routing["summaries"].items()
                                                  if names is None or k in names})
    return routing


def test_hash_projection_embedder_fails_the_gate(fresh, fast_probe):
    # THE HONEST OUTCOME, AND WHY: a hash projection is deterministic and self-consistent, but it is not in the
    # routing index's space (nomic, ABTT-corrected, 128-d), so each module's own summary self-recalls at about
    # chance (5/509). The gate refuses it -- as it must refuse a real CLM encoder for the same reason.
    m = _mind(url=fresh.url, emb_url=fresh.url + "/v1/embeddings")
    with pytest.raises(ValueError) as exc:
        m.set_embedder(m.clm_embedder())
    assert "space-agreement" in str(exc.value)
    assert any(p == "/v1/embeddings" for p, _, _ in fresh.seen)


def test_clm_raw_dimension_is_refused_before_the_space_probe(fresh, fast_probe):
    # CLM's real encoder returns 4096-d (Qwen3-8B); the index is 128-d. Refused at the first route() call.
    fresh.emb = "hash4096"
    m = _mind(url=fresh.url, emb_url=fresh.url + "/v1/embeddings")
    with pytest.raises(ValueError) as exc:
        m.set_embedder(m.clm_embedder())
    assert "4096" in str(exc.value)


def test_in_space_stub_passes_the_gate_through_the_plugin(fresh, fast_probe):
    # POSITIVE CONTROL: the same transport serving the index's own rows is admitted, so the refusals above are
    # the gate judging the SPACE, not the plugin failing to deliver vectors. normalize=False because the index's
    # ABTT correction is not scale-invariant; CLM's real embeddings arrive L2-normalised anyway.
    fresh.emb, fresh.oracle = "oracle", fast_probe["oracle"]
    m = _mind(url=fresh.url, emb_url=fresh.url + "/v1/embeddings")
    rep = m.set_embedder(m.clm_embedder(normalize=False))
    assert rep["ok"] is True and rep["rate"] >= 0.9
    m.set_embedder(None)


def test_embedder_validates_every_vector(fresh):
    m = _mind(url=fresh.url, emb_url=fresh.url + "/v1/embeddings")
    f = m.clm_embedder()
    v = f("pay my bill")
    assert v.shape == (128,) and abs(np.linalg.norm(v) - 1.0) < 1e-9
    assert f("pay my bill") is v                             # cached by text
    fresh.emb = "hash4096"
    with pytest.raises(clm_mod.CLMReplyError):
        f("a different text")                                # the dimension may not change between calls
    for bad in ("nan", "empty"):
        fresh.emb = bad
        with pytest.raises(clm_mod.CLMReplyError):
            m.clm_embedder()("text %s" % bad)
    with pytest.raises(clm_mod.CLMUnavailable):
        _mind(url=fresh.url).clm_embedder()                  # clm-serve alone exposes no vectors


def test_the_verbs_are_discoverable():
    m = _mind(_catalog=True)
    for query, verb in (("ask clm a typed question", "clm_systemone"),
                        ("rank candidate actions with clm", "clm_rank"),
                        ("is the clm server up", "clm_status")):
        top = [getattr(c, "name", "") for c in m.find_capability(query)[:3]]
        assert verb in top, (query, top)
