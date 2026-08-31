"""API LEARNING (cp66) -- extracted from leOS kernel_adapter_cert / api_adapter.

leOS's move, kept whole: LEARN an external API from its documentation (OpenAPI first,
no LLM anywhere in the parse), register its endpoints as callable tools, and -- the
part that makes it usable -- publish a DISCOVERABILITY CARD into the knowledge search
path, so "how do I get the weather" finds the tool the same way any other memory is
found. leOS wrote a KB article; here the card is a TEACH, which means the whole
existing contract applies for free: provenance, veto, session isolation, replay.

The call side builds the URL from the spec (path-param substitution, query params,
header auth), makes the request with stdlib urllib, parses JSON, and returns an honest
{ok, status, data | error}. Every successful call NOTES itself so usage becomes
experience (leOS records a displacement; we record through the same door).

HOSTED SAFETY, stated where it matters: arbitrary user-supplied URLs on a shared
server are an SSRF hole. The hosted zoo therefore only exposes APIs the OPERATOR
registered; per-user learning is a local-runtime feature.
"""
import json
import re
import urllib.request
import urllib.parse
import urllib.error

import numpy as np


class ApiToolbox:
    """Learn APIs from specs; call them by name; find them by task."""

    # NOT "__api_spec__": the cp38 control-token guard refuses standalone __token__
    # words on either side of a remember -- correctly, that is the attack class the
    # harsh battery taught it to refuse -- and it ate the first version of this
    # record. The registry rides as plain words instead; the guard stays untouched.
    SPEC_PREFIX = "api spec record: "

    def __init__(self, mind=None):
        self.mind = mind
        self.services = {}
        self._rehydrated = False

    def _rehydrate(self):
        """RESTART SURVIVAL (cp67, found by the battery): the discoverability cards
        persisted but the service registry died with the process -- after a restart
        every card pointed at an empty toolbox. Each learned spec is therefore
        TAUGHT as a record ("__api_spec__ <service>" -> spec json) and the toolbox
        rebuilds from those records lazily. Replay-safe by construction, and
        VETO-ABLE: vetoing a spec record un-registers the service, the same lever
        as everything else."""
        if self._rehydrated or self.mind is None:
            return
        self._rehydrated = True
        lad = self.mind.zoo.get("ladder")
        for t in getattr(lad, "taught_log", []):
            q = str(t[0])
            if q.startswith(self.SPEC_PREFIX) and (len(t) < 4 or
                                                   t[3] != "model-cached"):
                name = q[len(self.SPEC_PREFIX):].strip()
                try:
                    self.services[name] = json.loads(str(t[1]))
                except Exception:
                    pass

    # -- learning ---------------------------------------------------------
    def learn(self, spec, name=None, base_url=None, teach=True):
        """Ingest an OpenAPI spec (dict, JSON text, or a URL to fetch it from).
        Registers every operation; returns {service, endpoints}. With a mind
        attached and teach=True, each endpoint gets a discoverability card taught
        into memory -- contextual access is then ordinary recall."""
        if isinstance(spec, str) and spec.strip().startswith(("http://",
                                                              "https://")):
            with urllib.request.urlopen(spec, timeout=10) as r:
                spec = r.read().decode()
        if isinstance(spec, str):
            spec = json.loads(spec)
        title = name or re.sub(r"\W+", "_",
                               spec.get("info", {}).get("title", "api")).lower()
        base = base_url or (spec.get("servers") or [{}])[0].get("url", "")
        eps = {}
        for path, methods in (spec.get("paths") or {}).items():
            for method, op in methods.items():
                if method.upper() not in ("GET", "POST", "PUT", "DELETE"):
                    continue
                eid = op.get("operationId") or re.sub(
                    r"\W+", "_", "%s %s" % (method, path)).strip("_").lower()
                params = [{"name": p["name"], "in": p.get("in", "query"),
                           "required": bool(p.get("required"))}
                          for p in op.get("parameters", [])]
                eps[eid] = {"method": method.upper(), "path": path,
                            "params": params,
                            "description": op.get("summary") or
                            op.get("description") or eid}
        self.services[title] = {"base": base, "endpoints": eps}
        if self.mind is not None:
            self.mind.teach(self.SPEC_PREFIX + title,
                            json.dumps(self.services[title]))
        if teach and self.mind is not None:
            for eid, ep in eps.items():
                self.mind.teach(
                    "how do i %s" % ep["description"].lower().rstrip("."),
                    "use the learned api tool %s.%s -- %s %s%s with params %s; "
                    "call it via api_use(%r, %r, params={...})"
                    % (title, eid, ep["method"], base, ep["path"],
                       [p["name"] for p in ep["params"]], title, eid))
        return {"service": title, "base": base, "endpoints": sorted(eps)}

    # -- calling ----------------------------------------------------------
    def call(self, service, endpoint, params=None, headers=None, timeout=10):
        self._rehydrate()
        svc = self.services.get(service)
        if not svc or endpoint not in svc["endpoints"]:
            return {"ok": False, "error": "unknown %s.%s -- learned services: %s"
                    % (service, endpoint, sorted(self.services))}
        ep = svc["endpoints"][endpoint]
        params = dict(params or {})
        missing = [p["name"] for p in ep["params"]
                   if p.get("required") and p["name"] not in params]
        if missing:
            return {"ok": False, "error": "missing required param(s) %s for %s.%s"
                    % (missing, service, endpoint)}
        path = ep["path"]
        for p in ep["params"]:
            if p["in"] == "path" and p["name"] in params:
                path = path.replace("{%s}" % p["name"],
                                    urllib.parse.quote(str(params.pop(p["name"]))))
        url = svc["base"].rstrip("/") + path
        q = {k: v for k, v in params.items()}
        if ep["method"] == "GET" and q:
            url += "?" + urllib.parse.urlencode(q)
            body = None
        else:
            body = json.dumps(q).encode() if q else None
        req = urllib.request.Request(url, data=body, method=ep["method"],
                                     headers={"Content-Type": "application/json",
                                              **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode()
                try:
                    data = json.loads(raw)
                except Exception:
                    data = raw[:2000]
                out = {"ok": True, "status": r.status, "data": data}
        except urllib.error.HTTPError as e:
            raw = ""
            try:
                raw = e.read().decode()
                body = json.loads(raw)
            except Exception:
                body = raw[:500]
            return {"ok": False, "status": int(e.code), "data": body,
                    "error": "HTTP %d" % e.code}
        except Exception as e:
            return {"ok": False, "error": "%s: %s" % (type(e).__name__,
                                                      str(e)[:200])}
        if self.mind is not None:
            try:
                self.mind.drift_sentinel().note(
                    self.mind.semantic_key("%s.%s" % (service, endpoint))
                    ["vec"][:64],
                    self.mind.semantic_key(str(out["data"])[:200])["vec"][:64])
            except Exception:
                pass
        return out

    # -- contextual discovery --------------------------------------------
    def find(self, task, k=3):
        """Rank every learned endpoint against a task description -- semantic
        match plus the grounding doctrine (shared substantive token required)."""
        self._rehydrate()
        rows = []
        tt = {w for w in str(task).lower().split() if len(w) >= 4}
        for sname, svc in self.services.items():
            for eid, ep in svc["endpoints"].items():
                text = "%s %s %s" % (sname, eid, ep["description"])
                et = {w for w in text.lower().split() if len(w) >= 4}
                overlap = len(tt & et)
                if overlap:
                    rows.append({"tool": "%s.%s" % (sname, eid),
                                 "description": ep["description"][:100],
                                 "score": overlap})
        rows.sort(key=lambda r: -r["score"])
        return rows[:k]


def _selftest():
    """A REAL http loop, closed locally: serve a toy API in-process, learn it from
    its OpenAPI spec, discover the endpoint from a task phrase, call it over actual
    HTTP, and verify the discoverability card serves from memory."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/temp/"):
                city = self.path.split("/temp/")[1]
                body = json.dumps({"city": urllib.parse.unquote(city),
                                   "temp_c": 21.5}).encode()
            else:
                body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    spec = {"info": {"title": "toy weather"},
            "servers": [{"url": "http://127.0.0.1:%d" % port}],
            "paths": {"/temp/{city}": {"get": {
                "operationId": "get_temperature",
                "summary": "get the current temperature for a city",
                "parameters": [{"name": "city", "in": "path",
                                "required": True}]}}}}
    import lecore
    m = lecore.UnifiedMind()
    m.zoo_attach(lambda p: "")
    box = ApiToolbox(mind=m)
    rep = box.learn(spec)
    assert rep["endpoints"] == ["get_temperature"], rep
    hit = box.find("what is the temperature in a city")
    assert hit and hit[0]["tool"] == "toy_weather.get_temperature", hit
    r = box.call("toy_weather", "get_temperature", params={"city": "lisbon"})
    assert r["ok"] and r["data"]["temp_c"] == 21.5 and r["data"]["city"] == "lisbon"
    card = m.ask("how do i get the current temperature for a city")
    assert "toy_weather.get_temperature" in str(card.get("answer")), \
        "the discoverability card serves from ordinary memory (the leOS KB move)"
    bad = box.call("toy_weather", "nope")
    assert not bad["ok"] and "unknown" in bad["error"]
    miss = box.call("toy_weather", "get_temperature")
    assert not miss["ok"] and "missing required" in miss["error"]
    import tempfile as _tf, shutil as _sh
    _d = _tf.mkdtemp(prefix="apiln_")
    m.learning_save(_d)
    m2 = lecore.UnifiedMind()
    m2.zoo_attach(lambda p: "")
    m2.learning_load(_d)
    box2 = ApiToolbox(mind=m2)
    r2 = box2.call("toy_weather", "get_temperature", params={"city": "faro"})
    assert r2["ok"] and r2["data"]["city"] == "faro", \
        "the toolbox survives a restart by rehydrating from taught spec records"
    _sh.rmtree(_d, ignore_errors=True)
    srv.shutdown()
    return ("OK: learned an API from its spec with no LLM, found the endpoint from "
            "a task phrase, called it over real HTTP (127.0.0.1:%d, temp 21.5), and "
            "the discoverability card serves from memory with full provenance" % port)


if __name__ == "__main__":
    print(_selftest())
