"""THE LECORE CHAT (cp63) -- the front door, replacing the stale unified_app UI.

One page: talk to the substrate. By default there is NO model behind it -- answers come
from leCore memory (the bundled release memory, or your partition when one exists) with
provenance on every reply, escalation when it honestly does not know, and the void
explorer's conjectures attached when curiosity has something to offer. Creation runs
the same faculties the API exposes: describe a scene and the raymarcher renders it,
ask for a texture and the pattern algebra draws it, workspaces isolate what you teach.

Settings let you attach a model rung -- none (default), the local mini, or any
Ollama/OpenAI-compatible endpoint via LocalRung -- and the ladder's contract holds
regardless: taught facts serve from memory, model answers arrive marked model-cached,
vetoes stick.

Run:  python chat_server.py   (or run.bat / run.sh)  ->  http://127.0.0.1:7860
"""
import base64
import io
import json
import os
import re
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lecore

try:
    from flask import Flask, request, jsonify, Response
except ImportError:
    print("flask is required: pip install flask")
    raise

APP = Flask(__name__)
STATE = {"mind": None, "llm": "none", "llm_detail": "", "workspaces": {},
         "memories": {}, "conjectures": [], "caps": None,
         "lock": threading.Lock()}


def _caps():
    """Capability cards for 'how does X work' -- built once from the generated docs,
    so the chat explains the engine from the same source of truth the API ships."""
    if STATE["caps"] is None:
        cards = []
        here = os.path.dirname(os.path.abspath(__file__))
        for fn in ("docs/CAPABILITIES.md", "REFERENCE.md"):
            try:
                cur = None
                for line in open(os.path.join(here, fn)):
                    if line.startswith("#"):
                        if cur and len(cur[1]) > 60:
                            cards.append(cur)
                        cur = (line.strip("# \n"), "")
                    elif cur:
                        cur = (cur[0], (cur[1] + " " + line.strip())[:600])
                if cur and len(cur[1]) > 60:
                    cards.append(cur)
            except Exception:
                pass
        STATE["caps"] = cards
    return STATE["caps"]


def _mind():
    if STATE["mind"] is None:
        root = os.environ.get("LECORE_PARTITION")
        bundle = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "release_bundle")
        m = lecore.autoboot(partition=root, llm=None) if root else None
        if m is None:
            m = lecore.UnifiedMind()
            m.zoo_attach(lambda p: "")
            if os.path.isdir(bundle):
                try:
                    m.learning_load(bundle)
                except Exception:
                    pass
        STATE["mind"] = m
    return STATE["mind"]


def _png(arr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    a = np.clip(np.asarray(arr, float), 0, 1)
    plt.imsave(buf, a, format="png")
    return base64.b64encode(buf.getvalue()).decode()


def _swatch(stops):
    h, w = 48, 64 * len(stops)
    img = np.zeros((h, w, 3))
    for i, c in enumerate(stops):
        img[:, i * 64:(i + 1) * 64] = np.asarray(c)[:3]
    return img


def _texture(m, text, n=256):
    from holographic.misc.holographic_pattern import make_pattern, domain_warped_fbm
    tl = text.lower()
    if any(k in tl for k in ("wood", "marble", "smoke", "warp", "flow")):
        f = domain_warped_fbm(scale=3.0, warp=0.6, seed=7)
    else:
        kind = next((k for k in ("checker", "stripes", "dots", "fbm", "noise")
                     if k in tl), "fbm")
        f = make_pattern(kind, seed=7)
    xs = np.linspace(0, 2, n)
    P = np.stack(np.meshgrid(xs, xs, [0.5]), -1).reshape(-1, 3)
    v = np.asarray(f(P)).reshape(n, n)
    stops = np.asarray(m.palette_stops(seed=5, n=5))
    idx = np.clip((v * (len(stops) - 1)), 0, len(stops) - 1)
    lo = idx.astype(int)
    hi = np.minimum(lo + 1, len(stops) - 1)
    t = (idx - lo)[..., None]
    return stops[lo] * (1 - t) + stops[hi] * t


def _grounded(question, text):
    qt = {w for w in str(question).lower().split() if len(w) >= 4}
    at = {w for w in str(text).lower().split() if len(w) >= 4}
    return bool(qt & at) if qt else True


def _answer(m, msg):
    """Thin since cp68: the grounded answer routine moved DOWN into the engine
    (mind.ask_grounded) so raw api and hosted callers share the same protection;
    the chat keeps only the curiosity attachment."""
    g = m.ask_grounded(msg)
    cur = None
    if g["escalate"] and hasattr(m, "ask_curious"):
        try:
            c = m.ask_curious(msg)
            cur = c.get("curiosity") if isinstance(c, dict) else None
        except Exception:
            pass
    if g.get("provenance") == "model-cached" and STATE.get("llm") in (None,
                                                                      "none"):
        g = {"answer": "", "provenance": "escalated", "escalate": True}
    return g.get("answer", ""), bool(g.get("escalate")), g.get("provenance"), cur


def dispatch(msg, workspace):
    m = _mind()
    tl = msg.lower().strip()
    ws = None
    if workspace and workspace != "default":
        if workspace not in STATE["workspaces"]:
            STATE["workspaces"][workspace] = m.app_substrate("chat", workspace)
        ws = STATE["workspaces"][workspace]

    if tl in ("commands", "verbs", "cheatsheet"):
        return {"text": (
"Everything the chat understands (the api has more -- 'how does X work' explains any of it):\n"
"  TALK        just ask -- memory first with provenance; honest escalation\n"
"  teach: q = a          store a fact        |  wrong / veto: q   un-teach durably\n"
"  session: name         fresh context       |  health            memory saturation\n"
"  CREATE      render a scene: ... | show me a texture: ... | palette N\n"
"  MEMORIES    memories | load memory <path> | ask <slot>: q | compare: q\n"
"              export memory <dir>: <filter> | import memory <path> [theirs]\n"
"  DISCOVER    explore [: a, b, c] -> test N -> promote N   (the void loop)\n"
"              find a tool for <task> | how does <thing> work\n"
"  APIS        learn api: <openapi json> | use api: svc.endpoint {params}\n"
"  MODEL       settings (none / local runtime with auto-attribution / ollama)\n"
"Deeper instruments (api, one call each): signal_scan (windowed FDR scan), "
"solve_maze_grid, splat_memory, market_signal_test, pnp_restore, "
"model_attribute / ask_shortcut, attach_runtime, memory_list/export/import, "
"panel_seat/panel_deliberate, and the lean bridge (prove -> to_lean)."),
                "provenance": "engine"}

    if tl in ("health", "saturation", "status"):
        lad = m.zoo["ladder"]
        margs = getattr(lad, "_recent_margins", [])[-32:]
        sat = m.saturation_estimate(margs) if margs else {"state": "no-data"}
        return {"text": "Memory health: %s (margin %s) over %d taught rows. The "
                        "estimator is the cp59 result: recall MARGIN moves before "
                        "accuracy falls, so 'nearing-cliff' is an early warning, "
                        "not a post-mortem."
                        % (sat.get("state"), sat.get("mean_margin", "n/a"),
                           len(getattr(lad, "taught_log", []))),
                "provenance": "engine"}

    mt = re.match(r"(teach|remember)\s*:\s*(.+?)\s*=\s*(.+)$", msg, re.I | re.S)
    if mt:
        q, a = mt.group(2).strip(), mt.group(3).strip()
        warn = ""
        try:
            tc = m.teach_check(q, a)
            if tc.get("conflict_candidate"):
                warn = (" NOTE: the drift sentinel flags this as a possible "
                        "CONFLICT with established memory (verdict %s) -- it is "
                        "taught anyway; veto or re-teach decides."
                        % tc.get("verdict", "redshift"))
        except Exception:
            pass
        r = (ws.remember(q, a) if ws else m.teach(q, a))
        if isinstance(r, dict) and r.get("taught") is False:
            return {"text": "Not taught: %s" % r.get("reason", "refused")}
        return {"text": "Taught (%s): %r -> %r. It will serve with provenance "
                        "'taught' and a veto un-teaches it.%s"
                        % (workspace or "shared", q[:60], a[:60], warn)}

    if re.search(r"\b(render|draw|scene|model)\b", tl) and \
            not tl.startswith(("what", "how", "why")):
        desc = re.sub(r"^.*?(render|draw)( me)?( a)?\s*(scene\s*[:of]*\s*)?", "",
                      msg, flags=re.I) or msg
        try:
            cam = m.camera(eye=(0, 1.6, -4.2), target=(0, 0.6, 0), fov_deg=50,
                           aspect=4 / 3.0)
            img = m.render_scene_description(desc, cam, width=320, height=240,
                                             quality="fast")
            return {"text": "Rendered from the description (raymarched, "
                            "deterministic, no model involved):",
                    "artifact": _png(img), "caption": desc[:120]}
        except Exception as e:
            return {"text": "The scene parser could not build that (%s: %s). "
                            "Try naming shapes, colours, a floor, and lighting -- "
                            "e.g. 'a red sphere and a glass cube on a checker "
                            "floor at sunset'." % (type(e).__name__, str(e)[:120])}

    if re.search(r"\b(texture|pattern|material)\b", tl):
        try:
            return {"text": "A procedural texture from the pattern algebra "
                            "(seeded, reproducible):",
                    "artifact": _png(_texture(m, tl)), "caption": tl[:120]}
        except Exception as e:
            return {"text": "Texture path failed honestly: %s" % str(e)[:140]}

    if "palette" in tl:
        seed = int(re.search(r"\d+", tl).group()) if re.search(r"\d+", tl) else 5
        return {"text": "palette_stops(seed=%d):" % seed,
                "artifact": _png(_swatch(m.palette_stops(seed=seed, n=6)))}

    ms = re.match(r"session\s*:\s*(\S+)", tl)
    if ms:
        m.session_open(ms.group(1))
        return {"text": "Session %r opened -- teachings are tagged to it and replay "
                        "isolated, exactly like the API." % ms.group(1)}

    if tl in ("wrong", "veto", "no") or tl.startswith("veto:"):
        target = msg.split(":", 1)[1].strip() if ":" in msg else \
            STATE.get("last_q", "")
        if not target:
            return {"text": "Nothing to veto -- say  veto: <the question>  or reply "
                            "'wrong' right after an answer."}
        m.answer_feedback(target, ok=False)
        return {"text": "Vetoed %r. It stays dead across restarts (tombstoned); a "
                        "deliberate re-teach lifts it." % target[:70]}

    if tl == "memories":
        ml = m.memory_list()
        if not ml["memories"]:
            return {"text": "No named memories under %r yet. Export one with  "
                            "export memory <dir>: <filter>  or drop bundles "
                            "there." % ml["root"]}
        return {"text": "Named memories under %s:\n%s" % (ml["root"], "\n".join(
            "  %s  (%d KB)" % (e["name"], e["bytes"] // 1024)
            for e in ml["memories"]))}

    mx = re.match(r"export memory\s+(\S+)\s*(?::\s*(.+))?$", msg, re.I)
    if mx:
        rep = m.memory_export(mx.group(1), query=(mx.group(2) or None))
        return {"text": "Exported %d entries %s + %d tombstone(s) to %s -- "
                        "verified from a fresh boot: %s. Share the directory; "
                        "the importer gets the knowledge, the earned conjecture "
                        "rungs, the learned api tools, and your vetoes."
                        % (rep["exported"], rep["by_provenance"], rep["vetoes"],
                           rep["dest"], "clean" if rep["verified"] else
                           "%d MISS(ES) -- do not share" % rep["misses"])}

    mi = re.match(r"import memory\s+(\S+)\s*(theirs)?$", tl)
    if mi:
        imp = m.memory_import(mi.group(1),
                              on_conflict="theirs" if mi.group(2) else "flag")
        txt = ("Imported %d entr%s (%d identical skipped, %d veto(es) honored)."
               % (imp["imported"], "y" if imp["imported"] == 1 else "ies",
                  imp["skipped_identical"], imp["vetoes"]))
        if imp["conflicts"]:
            txt += "\nCONFLICTS (kept yours; say  import memory <path> theirs  "
            txt += "to adopt):"
            for c in imp["conflicts"][:5]:
                txt += "\n  %s\n    yours: %s\n    theirs: %s  [%s]" % (
                    c["q"][:60], c["mine"][:60], c["theirs"][:60], c["verdict"])
        return {"text": txt}

    mm = re.match(r"load memory\s+(.+)$", tl)
    if mm:
        path = mm.group(1).strip()
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            slot = lecore.UnifiedMind()
            slot.zoo_attach(lambda p: "")
            slot.learning_load(path if os.path.isdir(path)
                               else os.path.dirname(os.path.dirname(path)))
            STATE["memories"][name] = slot
            n = len([t for t in slot.zoo["ladder"].taught_log if len(t) > 3])
            return {"text": "Loaded memory %r (%d entries). Query it with  "
                            "ask %s: <question>   or   compare: <question>."
                            % (name, n, name)}
        except Exception as e:
            return {"text": "Could not load %r: %s" % (path, str(e)[:120])}

    ma = re.match(r"ask\s+(\w+)\s*:\s*(.+)$", msg, re.I | re.S)
    if ma and ma.group(1) in STATE["memories"]:
        slot = STATE["memories"][ma.group(1)]
        rr = slot.ask(ma.group(2).strip())
        sem = slot.recall_semantic(ma.group(2)) if hasattr(slot, "recall_semantic") \
            else {"found": False}
        ans = str(rr.get("answer") or "") or \
            (" / ".join(c["text"] for c in sem["candidates"][:2])
             if sem["found"] else "")
        return {"text": ans or "That memory holds nothing on this -- honest refusal.",
                "provenance": "%s:%s" % (ma.group(1),
                                         rr.get("provenance") or "recall")}

    mc = re.match(r"compare\s*:\s*(.+)$", msg, re.I | re.S)
    if mc and STATE["memories"]:
        qq = mc.group(1).strip()
        rows = []
        pool = {"current": m}
        pool.update(STATE["memories"])
        for name, slot in pool.items():
            sans, slow, _sp, _c = _answer(slot, qq)
            rows.append("%s -> %s" % (name, (sans if not slow
                                             else "(no grounded answer)")[:140]))
        return {"text": "The same question against every loaded memory -- where they "
                        "disagree is where teach_check and the drift sentinel earn "
                        "their keep:\n" + "\n".join(rows)}

    me = re.match(r"explore\s*(?::\s*(.+))?$", msg, re.I | re.S)
    if me:
        if me.group(1):
            items = [x.strip() for x in me.group(1).split(",") if x.strip()]
        else:
            lad = m.zoo["ladder"]
            items = sorted({str(t[0]) for t in lad.taught_log
                            if len(t) > 3 and t[3] in ("taught", "validated",
                                                       "evidenced")})[-24:]
        if len(items) < 4:
            return {"text": "Give me at least four concepts:  explore: a, b, c, d  "
                            "-- or teach a few facts first and just say  explore."}
        ex = m.explore(items, radius=0.5, budget=4)
        STATE["conjectures"] = ex["curious_about"]
        lines = []
        for i, c in enumerate(ex["curious_about"]):
            lines.append("%d. %s  x  %s   (p=%s, lens: %s)"
                         % (i + 1, c["a"][:44], c["b"][:44], c["p"],
                            (c.get("lens") or {}).get("nearest", "-")[:44]))
        return {"text": "The metaball map found %d collisions. The conjectures least "
                        "explainable by chance:\n%s\n\nSay  test 1  to run the "
                        "structural experiment on one, then  promote 1  if it "
                        "survives." % (ex["map"]["n_collisions"], "\n".join(lines)),
                "conjectures": [{"i": i + 1, "a": c["a"][:60], "b": c["b"][:60]}
                                for i, c in enumerate(ex["curious_about"])]}

    mtst = re.match(r"test\s+(\d+)", tl)
    if mtst and STATE["conjectures"]:
        i = int(mtst.group(1)) - 1
        if not (0 <= i < len(STATE["conjectures"])):
            return {"text": "No conjecture %d on the board." % (i + 1)}
        c = STATE["conjectures"][i]
        lad = m.zoo["ladder"]
        corpus = sorted({str(t[0]) for t in lad.taught_log if len(t) > 3})[-40:] or \
            [c["a"], c["b"]]
        mx = m.void_mix(c["a"], c["b"], corpus=corpus, null_trials=300)
        st = mx["structure"]
        # THE NULL IS THE REAL ONE: void_mix's actual 300 pairing draws, never a
        # synthetic distribution shaped around a mean -- a fabricated null is the
        # proxy trap wearing a lab coat.
        h = m.hypothesis_propose(
            "%s and %s share structure" % (c["a"][:40], c["b"][:40]),
            "the lens depth beats 300 chance pairings",
            lambda: {"measured": (mx["lens"] or {}).get("min_sim", 0.0),
                     "null": mx.get("null_draws") or []})
        v = m.hypothesis_test(h)
        STATE["conjectures"][i]["verdict"] = v
        return {"text": "EXPERIMENT on #%d: %s\n  lens depth %.3f vs null %.3f  "
                        "(p=%s over %d draws)\n  drift verdict: %s\n%s"
                        % (i + 1, v["status"].upper(),
                           (mx["lens"] or {}).get("min_sim", 0.0),
                           st["null_mean_lens"] or 0.0, v["p"], v["n_null"],
                           st["drift_verdict"],
                           "Say  promote %d  to record it as validated." % (i + 1)
                           if v["status"] == "pass" else
                           "It did not beat chance -- kept as a negative; that is "
                           "knowledge too."),
                "provenance": "experiment"}

    mp = re.match(r"promote\s+(\d+)", tl)
    if mp and STATE["conjectures"]:
        i = int(mp.group(1)) - 1
        c = STATE["conjectures"][i] if 0 <= i < len(STATE["conjectures"]) else None
        if not c or c.get("verdict", {}).get("status") != "pass":
            return {"text": "Only a conjecture that PASSED its experiment can be "
                            "promoted -- run  test %d  first." % (i + 1)}
        q = "conjecture: %s x %s" % (c["a"][:50], c["b"][:50])
        m.conjecture_record(q, "the pair shares structure; lens %r"
                            % (c.get("lens") or {}).get("nearest", ""))
        m.conjecture_promote(q, "validated",
                             "in-chat experiment p=%s" % c["verdict"]["p"])
        return {"text": "Promoted to VALIDATED, durably (survives restart, veto-able,"
                        " never served to taught_only callers). The research rung -- "
                        "a model or a human with the literature -- can raise it to "
                        "EVIDENCED.", "provenance": "validated"}

    ml = re.match(r"learn api\s*:\s*(.+)$", msg, re.I | re.S)
    if ml:
        try:
            rep = m.api_learn(ml.group(1).strip())
            return {"text": "Learned %r: endpoints %s. Each one taught a "
                            "discoverability card -- ask 'how do i ...' or "
                            "'find a tool for ...' to reach them; call with  "
                            "use api: service.endpoint {json params}"
                            % (rep["service"], ", ".join(rep["endpoints"]))}
        except Exception as e:
            return {"text": "Could not parse that spec: %s" % str(e)[:140]}
    mu = re.match(r"use api\s*:\s*(\w+)\.(\w+)\s*(\{.*\})?$", msg.strip(),
                  re.I | re.S)
    if mu:
        params = json.loads(mu.group(3)) if mu.group(3) else {}
        r = m.api_use(mu.group(1), mu.group(2), params=params)
        return {"text": json.dumps(r, indent=1)[:900],
                "provenance": "api-call" if r.get("ok") else "api-error"}
    mf = re.match(r"find (?:a |me a )?tool for (.+)$", tl)
    if mf:
        rows = m.tool_find(mf.group(1))
        if not rows:
            return {"text": "Nothing in the toolset matches that -- learn an api "
                            "(learn api: <openapi json>) or teach me about a tool."}
        return {"text": "Ranked against your task:\n" + "\n".join(
            "  %s -- %s" % (r["tool"], r["description"]) for r in rows)}
    mh = re.match(r"how does (.+?) work|explain (\w[\w\s]+)$", tl)
    if mh:
        topic = (mh.group(1) or mh.group(2)).strip()
        ex = m.explain(topic)
        if ex.get("found"):
            return {"text": "%s\n%s" % (ex["title"], ex["body"]),
                    "provenance": "docs"}

    if re.search(r"what is lecore|what can (you|it|lecore) do|^help$", tl):
        try:
            n = len(getattr(m, "catalog", lambda: [])()) or ""
        except Exception:
            n = ""
        return {"text": ("leCore is a deterministic holographic computing engine: "
                "meaning lives in high-dimensional vectors, structure is built by "
                "binding and superposition, and everything above that -- memory with "
                "provenance and honest escalation, a content-addressable image "
                "archive, procedural scenes/textures/meshes rendered by a real "
                "raymarcher, a slime-mould maze solver, drift detection, a void "
                "explorer that proposes and tests hypotheses -- is the same small "
                "algebra composed. Here: talk to it (answers come from leCore memory "
                "with provenance), teach it (teach: question = answer), or create -- "
                "try 'render a scene: ...', 'show me a texture: ...', 'palette 7'. "
                "Attach a model in settings only if you want one; the substrate "
                "never guesses."), "provenance": "engine"}
    r = (ws.recall(msg, established_only=False) if ws else None)
    if ws and r and (r.get("found") or r.get("answer")):
        return {"text": str(r.get("answer")), "provenance": "workspace"}
    STATE["last_q"] = msg
    ans, low, prov, cur = _answer(m, msg)
    out = {"text": ans if ans.strip() and not low else
           "I don't have that in memory. Teach me with  teach: question = answer,"
           " attach a model in Settings, or ask about the engine itself.",
           "provenance": prov or ("escalated" if low else None)}
    if cur and cur.get("conjectures"):
        out["curiosity"] = [{"a": c["a"][:60], "b": c["b"][:60],
                             "lens": (c.get("lens") or {}).get("nearest", "")[:60]}
                            for c in cur["conjectures"][:2]]
    return out


@APP.route("/")
def home():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "chat_ui.html")) as f:
        return Response(f.read(), mimetype="text/html")


@APP.route("/api/chat", methods=["POST"])
def chat():
    d = request.get_json(force=True)
    with STATE["lock"]:
        return jsonify(dispatch(str(d.get("message", "")),
                                str(d.get("workspace", "default"))))


@APP.route("/api/memory/upload", methods=["POST"])
def memory_upload():
    f = request.files["file"]
    name = os.path.splitext(f.filename)[0]
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "memories", name, "learning")
    os.makedirs(d, exist_ok=True)
    f.save(os.path.join(d, "state.lecore"))
    slot = lecore.UnifiedMind()
    slot.zoo_attach(lambda p: "")
    slot.learning_load(os.path.dirname(d))
    with STATE["lock"]:
        STATE["memories"][name] = slot
    n = len([t for t in slot.zoo["ladder"].taught_log if len(t) > 3])
    return jsonify({"name": name, "entries": n})


@APP.route("/api/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        d = request.get_json(force=True)
        choice = d.get("llm", "none")
        m = _mind()
        with STATE["lock"]:
            if choice == "none":
                m.zoo_attach(lambda p: "")
            elif choice == "mini":
                # cp71: the local model now runs on the engine's own RuntimeRung --
                # automatic source attribution (opt out with attribution off or
                # LECORE_NO_ATTRIBUTION=1), address-based shortcuts, stats.
                m.attach_runtime(d.get("path", "/tmp/mini_installed_full"),
                                 attribution=not d.get("no_attribution"))
            elif choice == "ollama":
                from tools.local_rung import LocalRung
                m.zoo_attach(LocalRung(backend="http",
                                       url=d.get("url",
                                                 "http://localhost:11434"),
                                       model=d.get("model", "qwen3.5:0.8")))
            STATE["llm"] = choice
            STATE["llm_detail"] = d.get("model", d.get("path", ""))
    return jsonify({"llm": STATE["llm"], "detail": STATE["llm_detail"],
                    "workspaces": ["default"] + sorted(STATE["workspaces"])})


if __name__ == "__main__":
    _mind()
    print("leCore chat -> http://127.0.0.1:7860  (substrate-only by default; "
          "attach a model in Settings)")
    APP.run(host="127.0.0.1", port=7860, debug=False)
