"""inv.py -- tiny client for the leCore service bus: inv(name, **args) -> result dict (errors returned, not raised)."""
import json, urllib.request, urllib.error
def inv(name, **args):
    req = urllib.request.Request("http://127.0.0.1:8080/invoke", data=json.dumps({"name": name, "args": args}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        return json.loads(urllib.request.urlopen(req, timeout=300).read()).get("result")
    except urllib.error.HTTPError as e:
        return {"HTTP": e.code, "body": e.read()[:300].decode()}
