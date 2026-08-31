"""Reach an LLM that lives in ANOTHER PROCESS -- the usual case, not the exception.

leCore's whole rung surface (attach_llm, agent_bridge, autoboot's llm=, agent_boot)
takes A LOCAL CALLABLE. That fits a model loaded in-process and nothing else. The
way people actually run this is Claude or ChatGPT behind an OpenAI-compatible
endpoint -- OpenWebUI, openzoo, ollama, LM Studio, a vendor API -- with the model
in a different process, often on a different machine.
Everything OpenAI-shaped in the tree pointed the OTHER WAY: unicron_serve_openai
puts a front door ON leCore so clients can call IN. Nothing called OUT.

So: a callable factory. `remote_llm(...)` returns exactly the `text -> text`
function every existing rung already accepts, which is why this needs no changes
anywhere else -- the seam was already the right shape, it just had nothing on the
far side.

stdlib only (urllib), per the constitution -- no requests, no vendor SDK.
"""

import json
import os
import urllib.error
import urllib.request


def remote_llm(url=None, model=None, api_key=None, timeout=60.0,
               max_tokens=512, temperature=0.0, extra=None):
    """A `text -> text` callable backed by an OpenAI-compatible /chat/completions.

    Every argument falls back to an environment variable, because the harness case
    is a container that sets env and runs a script -- nobody edits code to point at
    their gateway:
        url        LECORE_LLM_URL, OPENAI_BASE_URL   (default http://localhost:8402/v1,
                   the local openzoo proxy the integrations already assume)
        model      LECORE_LLM_MODEL, OPENAI_MODEL
        api_key    LECORE_LLM_KEY, OPENAI_API_KEY

    TEMPERATURE DEFAULTS TO 0.0, not the API's 1.0. A rung is part of a
    DETERMINISTIC engine; a sampling default would make the same question give
    different answers across runs and quietly break every reproducibility claim
    this repo makes. Raise it deliberately if you want variety.

    Raises on a failed call rather than returning an error string: a rung that
    returns "connection refused" as if it were an ANSWER poisons the trace and the
    taught store, and this engine already has abstain-not-error for the case where
    no answer exists.
    """
    base = (url or os.environ.get("LECORE_LLM_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "http://localhost:8402/v1").rstrip("/")
    mdl = (model or os.environ.get("LECORE_LLM_MODEL")
           or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini")
    key = (api_key or os.environ.get("LECORE_LLM_KEY")
           or os.environ.get("OPENAI_API_KEY") or "")
    endpoint = base + "/chat/completions"

    def call(prompt, **kw):
        body = {"model": kw.get("model", mdl),
                "messages": [{"role": "user", "content": str(prompt)}],
                "max_tokens": int(kw.get("max_tokens", max_tokens)),
                "temperature": float(kw.get("temperature", temperature))}
        if extra:
            body.update(extra)
        req = urllib.request.Request(
            endpoint, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        if key:
            req.add_header("Authorization", "Bearer " + key)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")[:200]
            except Exception:
                pass
            raise RuntimeError("remote_llm %s -> HTTP %s %s"
                               % (endpoint, e.code, detail)) from None
        except Exception as e:
            raise RuntimeError("remote_llm could not reach %s (%s). Set "
                               "LECORE_LLM_URL, or pass url=."
                               % (endpoint, type(e).__name__)) from None
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            # SHAPE, NOT GUESSWORK: report what came back instead of returning
            # None and letting the rung look like it answered with nothing.
            raise RuntimeError("remote_llm: unexpected response shape, keys=%r"
                               % (sorted(data) if isinstance(data, dict) else type(data),))

    call.__name__ = "remote_llm"
    call.endpoint = endpoint
    call.model = mdl
    return call


def _selftest():
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            seen["body"] = json.loads(self.rfile.read(n))
            seen["auth"] = self.headers.get("Authorization")
            out = json.dumps({"choices": [{"message": {"content": "pong"}}]})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out.encode())

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    fn = remote_llm(url="http://127.0.0.1:%d/v1" % port, model="m", api_key="k")
    assert fn("ping") == "pong"
    assert seen["body"]["messages"][0]["content"] == "ping", seen["body"]
    assert seen["body"]["temperature"] == 0.0, "a rung must default to deterministic"
    assert seen["auth"] == "Bearer k", seen["auth"]

    # a dead endpoint must RAISE, never return an error string as an answer
    dead = remote_llm(url="http://127.0.0.1:1/v1")
    try:
        dead("x")
        raise AssertionError("a dead endpoint returned instead of raising")
    except RuntimeError:
        pass
    srv.shutdown()
    print("holographic_remotellm selftest OK")


if __name__ == "__main__":
    _selftest()
