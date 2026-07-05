"""holographic_toolclient.py -- call another node the same way leCore is called.

The other half of the tool interface. A leCore service exposes GET /tools (a manifest) and POST /invoke ({name, args}
-> result). This module lets THIS process consume that: fetch a remote node's /tools and wrap each as a callable, so a
remote faculty is used exactly like a local one -- and can be registered into the orchestrator to plan alongside local
tools, remote tools, LLMs, and shell commands uniformly.

The node on the other end can be another leCore instance, or any harness that speaks the same two endpoints. stdlib
urllib only -- no HTTP client dependency.

    from holographic_toolclient import remote_tools
    for t in remote_tools("http://other-node:8080", token="secret"):
        print(t.name, "-", t.description)
        result = t.run({"vec_a": [1, 0], "vec_b": [0, 1]})     # POSTs to the remote /invoke and returns the result

Or one-shot:  call("http://other-node:8080", "opponent_channels", {"vec_a": [...], "vec_b": [...]}, token="secret")
"""
import json
import urllib.request
import urllib.error


class RemoteTool:
    """One tool discovered on a remote node. `run(args)` POSTs {name, args} to the node's /invoke and returns the
    result. `name`/`description`/`params` come from the remote manifest. Registerable into the orchestrator."""

    def __init__(self, base_url, name, description="", params=None, token=None, timeout=30.0):
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.description = description
        self.params = params or []
        self.token = token
        self.timeout = timeout

    def run(self, args=None):
        """Invoke the remote tool with `args` (a dict of keyword arguments). Returns the remote result, or raises."""
        resp = _post(self.base_url + "/invoke", {"name": self.name, "args": args or {}},
                     token=self.token, timeout=self.timeout)
        if not resp.get("ok", False):
            raise RuntimeError("remote %s failed: %s" % (self.name, resp.get("error", "unknown error")))
        return resp.get("result")

    # so it drops into the orchestrator, which looks for .fn(value)
    def fn(self, args=None):
        return self.run(args)

    def __repr__(self):
        return "RemoteTool(%r @ %s)" % (self.name, self.base_url)


def remote_tools(base_url, token=None, timeout=30.0):
    """Fetch a remote node's /tools manifest and yield each as a RemoteTool whose run() POSTs to that node's /invoke.
    Now a Planner (or you) can chain local faculties and remote tools uniformly."""
    manifest = _get(base_url.rstrip("/") + "/tools", token=token, timeout=timeout)
    for spec in manifest.get("tools", []):
        yield RemoteTool(base_url, spec["name"], spec.get("description", ""),
                         spec.get("params", []), token=token, timeout=timeout)


def call(base_url, name, args=None, token=None, timeout=30.0):
    """One-shot: invoke a single named tool on a remote node and return its result (no manifest fetch)."""
    return RemoteTool(base_url, name, token=token, timeout=timeout).run(args)


def list_tools(base_url, token=None, timeout=30.0):
    """The remote manifest as a list of {name, description, params} dicts (what GET /tools returns)."""
    return _get(base_url.rstrip("/") + "/tools", token=token, timeout=timeout).get("tools", [])


# ---- tiny stdlib HTTP helpers (no third-party client) ------------------------------------------------------
def _headers(token):
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = "Bearer %s" % token                # the service's bearer-token auth
    return h


def _get(url, token=None, timeout=30.0):
    req = urllib.request.Request(url, headers=_headers(token), method="GET")
    return _send(req, timeout)


def _post(url, body, token=None, timeout=30.0):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=_headers(token), method="POST")
    return _send(req, timeout)


def _send(req, timeout):
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:                          # a 4xx/5xx still carries a JSON error body
        try:
            return json.loads(e.read().decode("utf-8"))
        except Exception:
            raise
