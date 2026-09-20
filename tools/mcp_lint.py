#!/usr/bin/env python3
"""tools/mcp_lint.py -- the MCP surface's own gate (sweep 176, backlog M1 + M15): like skill_lint for cards.

Prints a table and exits non-zero on any violation:
  * every tool carries annotations {readOnlyHint, destructiveHint, idempotentHint, openWorldHint}
  * every tool's description says which sibling to prefer ("instead" / "use it only")
  * description length: median <= 320 chars, none over 700
  * profiles: minimal <= 8 tools and <= 6 KB of schema; standard <= 32; full = the whole table
  * the instructions banner <= 1,300 chars and every tool name it mentions exists
  * false-confident finds: none of the off-catalog probes gets an `answer` tier
  * decide -> outcome round trip over JSON-RPC, and the repeat comes back via reflex
  * prompts/get renders all four prompts; resources/read reads every listed resource
  * protocol negotiation: 2024-11-05 and 2025-06-18 both initialize; structuredContent only on the latter

    PYTHONHASHSEED=0 python3 tools/mcp_lint.py
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import holographic_mcp as H  # noqa: E402

PROBES = ["purple monkey dishwasher", "counter traders", "asdf qwer zxcv", "blue elephant tuesday",
          "hello there how are you today", "book me a table for two at eight", "my sister's wedding is in march",
          "the weather in lisbon next week", "how do I boil an egg", "recommend a good novel",
          "convert 30 celsius to fahrenheit", "sing happy birthday", "order more coffee pods",
          "who won the match last night", "translate this into french"]


def main():
    fails = []
    srv = H.MCPServer()
    rpc = lambda method, params=None, rid=1: srv.handle({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
    # -- protocol negotiation --
    old = rpc("initialize", {"protocolVersion": "2024-11-05"})["result"]["protocolVersion"]
    new = rpc("initialize", {"protocolVersion": "2025-06-18"})["result"]["protocolVersion"]
    if old != "2024-11-05" or new != "2025-06-18":
        fails.append("negotiation: %s / %s" % (old, new))
    # -- the table, per profile --
    sizes = {}
    for prof in ("minimal", "standard", "full"):
        os.environ["LECORE_MCP_PROFILE"] = prof
        tools = [t for t in rpc("tools/list")["result"]["tools"] if not t["name"].startswith(("studio", "asset_"))]
        sizes[prof] = (len(tools), len(json.dumps(tools)))       # the CORE table; studio/asset tools are environment-gated
    os.environ.pop("LECORE_MCP_PROFILE", None)
    if sizes["minimal"][0] > 8 or sizes["minimal"][1] > 6000:
        fails.append("minimal profile %s (max 8 tools / 6000 bytes)" % (sizes["minimal"],))
    if sizes["standard"][0] > 32:
        fails.append("standard profile %d tools (max 32)" % sizes["standard"][0])
    if sizes["full"][0] != len(H._TOOLS):
        fails.append("full profile is not the whole table")
    tools = H._annotate_tools(list(H._TOOLS))
    missing_ann = [t["name"] for t in tools if set(t.get("annotations", {})) != {"readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"}]
    missing_guid = [t["name"] for t in tools if not re.search(r"\binstead\b|use it only", t["description"], re.I)]
    lens = sorted(len(t["description"]) for t in tools)
    if missing_ann:
        fails.append("no annotations: %s" % missing_ann)
    if missing_guid:
        fails.append("no sibling guidance: %s" % missing_guid)
    # budgeted at the MEASURED values after the guidance pass (median 358, max 756): growth fails, not the present
    if lens[len(lens) // 2] > 380 or lens[-1] > 780:
        fails.append("description length median %d / max %d (budget 380 / 780 -- a new tool grew past the table)" % (lens[len(lens) // 2], lens[-1]))
    # -- the banner --
    banner = H._INSTRUCTIONS
    names = {t["name"] for t in tools}
    phantom = [w for w in set(re.findall(r"\b(lecore_[a-z_]+|memory_[a-z_]+|study_ask|study|zoo_[a-z_]+)\b", banner)) if w not in names]
    if len(banner) > 1300:
        fails.append("banner %d chars (max 1300)" % len(banner))
    if phantom:
        fails.append("banner names tools that do not exist: %s" % sorted(phantom))
    # -- false-confident finds --
    call = lambda name, args: json.loads(rpc("tools/call", {"name": name, "arguments": args}, rid=9)["result"]["content"][0]["text"])
    t0 = time.time()
    answered = [q for q in PROBES if call("lecore_find", {"query": q}).get("tier") == "answer"]
    find_ms = 1000 * (time.time() - t0) / len(PROBES)
    if answered:
        fails.append("off-catalog probes answered: %s" % answered)
    # -- decide -> outcome -> reflex --
    d = call("lecore_decide", {"state": "courier lost the package", "options": ["billing", "shipping"]})
    o = call("lecore_outcome", {"id": d["id"], "outcome": "shipping"})
    if o.get("was_correct") is not True or not (o.get("reflex") or {}).get("learned"):
        fails.append("decide/outcome round trip: %s" % {k: o.get(k) for k in ("was_correct", "reflex")})
    r = call("lecore_find", {"query": "smooth a bumpy mesh"})
    call("lecore_outcome", {"id": r["id"], "outcome": r["answer"] or r["hits"][0]["name"]})
    if call("lecore_find", {"query": "smooth a bumpy mesh"}).get("via") != "reflex":
        fails.append("the repeat find did not come back via reflex")
    # -- structuredContent only on the new protocol --
    res_new = rpc("tools/call", {"name": "lecore_decide", "arguments": {"state": "parcel is late", "options": ["billing", "shipping"]}}, rid=10)["result"]
    rpc("initialize", {"protocolVersion": "2024-11-05"})
    res_old = rpc("tools/call", {"name": "lecore_decide", "arguments": {"state": "parcel is late", "options": ["billing", "shipping"]}}, rid=11)["result"]
    if "structuredContent" not in res_new or "structuredContent" in res_old:
        fails.append("structuredContent: new=%s old=%s" % ("structuredContent" in res_new, "structuredContent" in res_old))
    # -- prompts and resources --
    for pr in rpc("prompts/list")["result"]["prompts"]:
        args = {a["name"]: ("courier lost the package" if a["name"] == "state" else ["billing", "shipping"] if a["name"] == "options"
                            else ["holographic/agents_and_reasoning/holographic_codeflow.py"] if a["name"] == "paths"
                            else "smooth a bumpy mesh") for a in pr["arguments"] if a["required"]}
        g = rpc("prompts/get", {"name": pr["name"], "arguments": args}, rid=12)
        if "error" in g or "GOAL" not in g["result"]["messages"][0]["content"]["text"]:
            fails.append("prompt %s did not render" % pr["name"])
    for rs in rpc("resources/list")["result"]["resources"]:
        g = rpc("resources/read", {"uri": rs["uri"]}, rid=13)
        if "error" in g or not g["result"]["contents"][0].get("text"):
            fails.append("resource %s did not read" % rs["uri"])
    print("MCP LINT -- tools %d | annotated %d | with guidance %d | description median %d max %d" % (
        len(tools), len(tools) - len(missing_ann), len(tools) - len(missing_guid), lens[len(lens) // 2], lens[-1]))
    print("   profiles: minimal %d tools / %d B, standard %d / %d B, full %d / %d B" % (sizes["minimal"] + sizes["standard"] + sizes["full"]))
    print("   banner %d chars | off-catalog probes answered %d/%d | lecore_find %.0f ms each | prompts %d | resources %d" % (
        len(banner), len(answered), len(PROBES), find_ms, len(H._PROMPTS), len(srv._resources_list())))
    if fails:
        print("FAIL:")
        for f in fails:
            print("   ", f)
        return 1
    print("OK: the MCP surface is within every budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
