"""bench_poster_swarm.py -- four workers paint the sweep-176 benchmark poster in leStudio.

Transport is leCore's: the plan and every worker step go on the service bus (POST /invoke swarm_step --
refused without done_when + evidence), the region map and palette live in the shared memory partition
(teach/ask), design choices are TYPED DECISIONS through systemone (reflex on: a repeat of the same
brief later is answered from experience), and every worker verifies its own region with /api/analyze
before it reports. Workers are sequential here (one core); each has its own X-User / X-Client.

    PYTHONHASHSEED=0 python3 bench_poster_swarm.py
"""
import json
import math
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont

STUDIO = "http://127.0.0.1:5050"
CORE = "http://127.0.0.1:8094"
W, HGT = 1600, 1000
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LABELS = "/home/claude/poster_labels.png"


def core(name, args):
    req = urllib.request.Request(CORE + "/invoke", data=json.dumps({"name": name, "args": args}).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    if "result" not in d:
        raise SystemExit("core %s: %s" % (name, str(d)[:300]))
    return d["result"]


def bus(topic, payload):
    req = urllib.request.Request(CORE + "/bus/publish", data=json.dumps({"topic": topic, "from": payload.get("worker", "orchestrator"), "payload": payload}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


class Worker:
    """One studio identity. Every POST carries a stable X-User and a per-run X-Client."""

    def __init__(self, name):
        self.name = name
        self.H = {"Content-Type": "application/json", "X-Client": "%s-%d" % (name, time.time()), "X-User": name}
        self.strokes = []

    def post(self, path, body):
        req = urllib.request.Request(STUDIO + path, data=json.dumps(body).encode(), headers=self.H, method="POST")
        try:
            return json.loads(urllib.request.urlopen(req, timeout=600).read())
        except urllib.error.HTTPError as e:
            raise SystemExit("HTTP %d on %s: %s" % (e.code, path, e.read()[:400]))

    def get(self, path):
        return urllib.request.urlopen(STUDIO + path, timeout=600).read()

    def stroke(self, pts, color, radius=1.5, opacity=1.0, layer=None):
        self.strokes.append({"layer": layer, "points": [[float(x), float(y)] for x, y in pts], "color": color,
                             "radius": radius, "opacity": opacity, "hardness": 1.0, "record": True})

    def seg(self, p0, p1, n=8):
        return [[p0[0] + (p1[0] - p0[0]) * i / (n - 1), p0[1] + (p1[1] - p0[1]) * i / (n - 1)] for i in range(n)]

    def line(self, p0, p1, **kw):
        self.stroke(self.seg(p0, p1), **kw)

    def rect(self, x0, y0, x1, y1, **kw):
        pts = self.seg([x0, y0], [x1, y0]) + self.seg([x1, y0], [x1, y1])[1:] + self.seg([x1, y1], [x0, y1])[1:] + self.seg([x0, y1], [x0, y0])[1:]
        self.stroke(pts, **kw)

    def fill_rect(self, x0, y0, x1, y1, color, layer, step=3.0, opacity=1.0):
        """A filled bar is a comb of horizontal strokes -- the studio deposits paint, it does not fill polygons."""
        y = y0
        while y <= y1:
            self.line([x0, y], [x1, y], color=color, radius=step * 0.75, opacity=opacity, layer=layer)
            y += step

    def flush(self):
        n = 0
        for i in range(0, len(self.strokes), 400):
            r = self.post("/api/paint_batch", {"strokes": self.strokes[i:i + 400]})
            n += len(self.strokes[i:i + 400])
        self.strokes.clear()
        return n

    def analyze(self, regions):
        return self.post("/api/analyze", {"regions": regions})

    def step(self, state, tool, done_when, evidence, args=None):
        """The swarm step contract: refused by the mind without done_when and evidence."""
        return core("swarm_step", {"state": state, "tool": tool, "args": args or {}, "done_when": done_when,
                                   "evidence": evidence, "worker": self.name, "topic": "bench-poster"})


# ---- the numbers this poster shows (this session's measurements; the source of each in the label) ----------
DATA = {
    "both_ends": [(0.9, 32.9, 0.964), (0.8, 51.2, 0.908), (0.7, 62.9, 0.865), (0.5, 79.8, 0.808)],   # p floor, kept %, acc
    "published": [("10-shot", 85.2), ("30-shot", 90.6)],
    "gate": [("old gate\naccepts", 4.7), ("tiered:\nanswer", 32.0), ("menu", 31.6), ("refuse", 36.4)],
    "reflex": [("repeat", 99.3, 0.982), ("new paraphrase", 20.9, 0.810)],
    "conformal": [("nb 0.95", 0.960), ("proto 0.95", 0.961), ("nb 0.90", 0.913), ("proto 0.90", 0.918)],
}

# ================================ 0. orchestrator: decisions, memory, plan ==================================
t0 = time.time()
brief = "a dense dark-ground technical poster of measured numbers, four panels, monospace labels, one accent colour"
decisions = core("systemone_decide", {"state": brief, "questions": {
    "ground": {"type": "choice", "options": ["dark", "light"], "examples": {
        "dark": ["dark ground with light ink", "navy background bright labels", "a night poster, white text on black", "dense dark technical sheet"],
        "light": ["white paper black ink", "a light airy page", "bright ground dark type", "printed white sheet"]}},
    "accent": {"type": "choice", "options": ["amber", "cyan"], "examples": {
        "amber": ["warm accent on a dark ground", "orange highlight bars", "amber numbers", "warm single accent colour"],
        "cyan": ["cool accent", "cyan highlight on navy", "electric blue accent", "cold single accent colour"]}}},
    "scorer": "nb", "encoder": "ngram", "margin": 0.0, "reflex": True})
ground = decisions["ground"]["value"] or "dark"
accent = decisions["accent"]["value"] or "amber"
print("typed decisions:", {k: (v.get("value"), v.get("via", "typed")) for k, v in decisions.items()})
PAL = {"dark": {"bg": [0.06, 0.07, 0.10], "ink": [0.92, 0.93, 0.95], "faint": [0.30, 0.33, 0.40], "panel": [0.10, 0.12, 0.17]},
       "light": {"bg": [0.96, 0.96, 0.94], "ink": [0.08, 0.08, 0.10], "faint": [0.65, 0.66, 0.70], "panel": [0.90, 0.90, 0.88]}}[ground]
PAL["accent"] = {"amber": [1.0, 0.68, 0.18], "cyan": [0.25, 0.85, 1.0]}[accent]
PAL["accent2"] = [0.55, 0.60, 0.70]
regions = {"A": [60, 130, 780, 540], "B": [820, 130, 1540, 540], "C": [60, 580, 780, 940], "D": [820, 580, 1540, 940]}
core("teach", {"query": "bench poster: region map", "answer": json.dumps(regions)})
core("teach", {"query": "bench poster: palette", "answer": json.dumps(PAL)})
bus("bench-poster", {"plan": ["W1 layout: ground, panel frames, grid", "W2 bars: panels A (both ends) and B (gate)",
                              "W3 curves: risk-coverage line in A, reflex + conformal in C and D", "W4 labels: title, axes, numbers"],
                     "done_when": "every panel region brighter than the ground by /api/analyze; labels landed; .lews saved",
                     "decisions": {"ground": ground, "accent": accent}})

# ================================ 1. W1 layout =============================================================
w1 = Worker("w1-layout")
w1.post("/api/new", {"name": "bench_poster_sweep176", "width": W, "height": HGT, "background": PAL["bg"], "dpi": 120})
layers = {}
for nm in ("frames", "bars", "curves"):
    w1.post("/api/layer", {"action": "add", "name": nm})
    st = json.loads(w1.get("/api/state"))
    layers[nm] = [l["id"] for l in st["layers"] if l.get("name") == nm][-1]
core("teach", {"query": "bench poster: layer ids", "answer": json.dumps(layers)})
reg = json.loads(core("ask", {"query": "bench poster: region map"})["answer"])          # the worker reads the shared memory
for k, (x0, y0, x1, y1) in reg.items():
    w1.fill_rect(x0, y0, x1, y1, PAL["panel"], layers["frames"], step=4.0)
    w1.rect(x0, y0, x1, y1, color=PAL["faint"], radius=1.4, layer=layers["frames"])
    for i in range(1, 5):                                                                  # a faint horizontal grid per panel
        y = y0 + (y1 - y0) * i / 5
        w1.line([x0 + 8, y], [x1 - 8, y], color=PAL["faint"], radius=0.7, opacity=0.35, layer=layers["frames"])
w1.line([60, 100], [1540, 100], color=PAL["accent"], radius=2.0, layer=layers["frames"])
n = w1.flush()
a = w1.analyze([{"name": k, "box": [x0 / W, y0 / HGT, x1 / W, y1 / HGT], "metrics": ["brightness"]} for k, (x0, y0, x1, y1) in reg.items()]
                + [{"name": "ground", "box": [0.01, 0.95, 0.20, 0.99], "metrics": ["brightness"]}])
br = {r["name"]: r.get("brightness") for r in a.get("regions", a if isinstance(a, list) else [])}
w1.step("lay out four panels on the ground", "paint_batch", "every panel brighter than the ground", {"strokes": n, "brightness": br})
print("W1 layout:", n, "strokes | panel brightness", {k: round(v, 3) for k, v in br.items() if v is not None})

# ================================ 2. W2 bars ===============================================================
w2 = Worker("w2-bars")
reg = json.loads(core("ask", {"query": "bench poster: region map"})["answer"])
pal = json.loads(core("ask", {"query": "bench poster: palette"})["answer"])
lay = json.loads(core("ask", {"query": "bench poster: layer ids"})["answer"])
# panel A: kept-by-substrate bars at each p floor + the published few-shot marks as thin bars
x0, y0, x1, y1 = reg["A"]; base = y1 - 40; top = y0 + 70; scale = (base - top) / 100.0
slots = len(DATA["both_ends"]) + len(DATA["published"]); bw = (x1 - x0 - 80) / slots * 0.6
for i, (p, kept, acc) in enumerate(DATA["both_ends"]):
    cx = x0 + 60 + (x1 - x0 - 80) * (i + 0.5) / slots
    w2.fill_rect(cx - bw / 2, base - kept * scale, cx + bw / 2, base, pal["accent"], lay["bars"])
    w2.fill_rect(cx - bw / 2, base - acc * 100 * scale, cx - bw / 2 + 6, base, pal["ink"], lay["bars"])   # accuracy tick on the left edge
for j, (nm, val) in enumerate(DATA["published"]):
    cx = x0 + 60 + (x1 - x0 - 80) * (len(DATA["both_ends"]) + j + 0.5) / slots
    w2.fill_rect(cx - bw / 2, base - val * scale, cx + bw / 2, base, pal["accent2"], lay["bars"])
# panel B: the gate
x0, y0, x1, y1 = reg["B"]; base = y1 - 40; top = y0 + 70; scale = (base - top) / 100.0
slots = len(DATA["gate"]); bw = (x1 - x0 - 80) / slots * 0.55
for i, (nm, val) in enumerate(DATA["gate"]):
    cx = x0 + 60 + (x1 - x0 - 80) * (i + 0.5) / slots
    w2.fill_rect(cx - bw / 2, base - val * scale, cx + bw / 2, base, pal["accent"] if i else pal["accent2"], lay["bars"])
n = w2.flush()
a = w2.analyze([{"name": "A_bars", "box": [reg["A"][0] / W, (reg["A"][1] + 70) / HGT, reg["A"][2] / W, (reg["A"][3] - 40) / HGT], "metrics": ["brightness", "dominant_hue"]},
                {"name": "B_bars", "box": [reg["B"][0] / W, (reg["B"][1] + 70) / HGT, reg["B"][2] / W, (reg["B"][3] - 40) / HGT], "metrics": ["brightness"]}])
ev = {r["name"]: {k: r.get(k) for k in ("brightness", "dominant_hue")} for r in a.get("regions", [])}
w2.step("bars for panels A and B from the measured numbers", "paint_batch", "bar regions brighter than the empty panel; accent hue present", {"strokes": n, **ev})
print("W2 bars:", n, "strokes |", {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in ev.items()})

# ================================ 3. W3 curves =============================================================
w3 = Worker("w3-curves")
reg = json.loads(core("ask", {"query": "bench poster: region map"})["answer"]); pal = json.loads(core("ask", {"query": "bench poster: palette"})["answer"]); lay = json.loads(core("ask", {"query": "bench poster: layer ids"})["answer"])
# panel A: risk-coverage as a line over the bars (kept% on x, accuracy on y)
x0, y0, x1, y1 = reg["A"]
pts = [[x0 + 60 + (x1 - x0 - 120) * kept / 100.0, y1 - 40 - (y1 - y0 - 110) * acc] for _, kept, acc in DATA["both_ends"]]
w3.stroke(w3.seg(pts[0], pts[1]) + w3.seg(pts[1], pts[2])[1:] + w3.seg(pts[2], pts[3])[1:], color=pal["ink"], radius=2.4, layer=lay["curves"])
for p in pts:
    w3.stroke([[p[0] + 7 * math.cos(t), p[1] + 7 * math.sin(t)] for t in [2 * math.pi * i / 24 for i in range(25)]], color=pal["ink"], radius=2.0, layer=lay["curves"])
# panel C: reflex learning -- two bars for fire rate, accuracy as a tick
x0, y0, x1, y1 = reg["C"]; base = y1 - 40; top = y0 + 70; scale = (base - top) / 100.0
for i, (nm, fired, acc) in enumerate(DATA["reflex"]):
    cx = x0 + 60 + (x1 - x0 - 80) * (i + 0.5) / 2; bw = (x1 - x0 - 80) / 2 * 0.45
    w3.fill_rect(cx - bw / 2, base - fired * scale, cx + bw / 2, base, pal["accent"], lay["curves"])
    w3.line([cx - bw / 2 - 12, base - acc * 100 * scale], [cx + bw / 2 + 12, base - acc * 100 * scale], color=pal["ink"], radius=2.2, layer=lay["curves"])
# panel D: conformal coverage -- nominal as a faint line, empirical as bars
x0, y0, x1, y1 = reg["D"]; base = y1 - 40; top = y0 + 70; scale = (base - top) / 1.0
for i, (nm, cov) in enumerate(DATA["conformal"]):
    cx = x0 + 60 + (x1 - x0 - 80) * (i + 0.5) / 4; bw = (x1 - x0 - 80) / 4 * 0.5
    nominal = 0.95 if "0.95" in nm else 0.90
    w3.fill_rect(cx - bw / 2, base - cov * scale, cx + bw / 2, base, pal["accent"], lay["curves"])
    w3.line([cx - bw / 2 - 14, base - nominal * scale], [cx + bw / 2 + 14, base - nominal * scale], color=pal["ink"], radius=2.0, layer=lay["curves"])
n = w3.flush()
a = w3.analyze([{"name": "C", "box": [reg["C"][0] / W, (reg["C"][1] + 70) / HGT, reg["C"][2] / W, (reg["C"][3] - 40) / HGT], "metrics": ["brightness"]},
                {"name": "D", "box": [reg["D"][0] / W, (reg["D"][1] + 70) / HGT, reg["D"][2] / W, (reg["D"][3] - 40) / HGT], "metrics": ["brightness"]}])
ev = {r["name"]: r.get("brightness") for r in a.get("regions", [])}
w3.step("risk-coverage line in A; reflex and conformal bars in C and D", "paint_batch", "C and D regions brighter than the empty panel", {"strokes": n, "brightness": ev})
print("W3 curves:", n, "strokes |", {k: round(v, 3) for k, v in ev.items() if v is not None})

# ================================ 4. W4 labels =============================================================
w4 = Worker("w4-labels")
reg = json.loads(core("ask", {"query": "bench poster: region map"})["answer"])
img = Image.new("RGB", (W, HGT), (0, 0, 0)); d = ImageDraw.Draw(img)
ink = tuple(int(255 * c) for c in PAL["ink"]); acc = tuple(int(255 * c) for c in PAL["accent"]); dim = tuple(int(255 * c) for c in PAL["faint"])
def T(x, y, s, sz=16, c=ink, bold=False, anchor="la"):
    d.text((x, y), s, fill=c, font=ImageFont.truetype(FONTB if bold else FONT, sz), anchor=anchor)
OVERFLOW = []                              # every text box is measured against its panel before baking
def fits(x, y, s, sz, box, bold=False, anchor="la"):
    """textbbox in the SAME font/anchor the draw uses; a box outside `box` is an overflow (evidence, not a hope)."""
    bb = d.textbbox((x, y), s, font=ImageFont.truetype(FONTB if bold else FONT, sz), anchor=anchor)
    ok = bb[0] >= box[0] - 2 and bb[2] <= box[2] + 2 and bb[1] >= box[1] - 2 and bb[3] <= box[3] + 2
    if not ok:
        OVERFLOW.append((s[:30], [round(v) for v in bb], box))
    return ok
def wrap_to(s, sz, width, bold=False):
    """Greedy word wrap by measured pixel width -- long panel titles ran into the neighbouring panel."""
    words, lines, cur = s.split(), [], ""
    f = ImageFont.truetype(FONTB if bold else FONT, sz)
    for w_ in words:
        t = (cur + " " + w_).strip()
        if d.textlength(t, font=f) <= width:
            cur = t
        else:
            lines.append(cur); cur = w_
    return lines + [cur]
PAGE = [0, 0, W, HGT]
T(60, 34, "leCore sweep 176 -- measured, not promised", 34, bold=True); fits(60, 34, "leCore sweep 176 -- measured, not promised", 34, PAGE, bold=True)
sub = "Banking77 20/intent, AG News, ablated aliases; 3 seeds unless stated  |  19 Sep 2026"
T(60, 78, sub, 14, c=dim); fits(60, 78, sub, 14, [0, 70, W, 100])
titles = {"A": "A. both ends: what the substrate keeps (bars) at each p floor, its accuracy (ticks/line); grey = published fine-tuned few-shot",
          "B": "B. the gate: the old null gate accepted 4.7% of honest paraphrases; tiered routing answers / menus / refuses",
          "C": "C. the reflex arc learning from use: fired (bar) and accuracy (tick) on repeats vs new paraphrases",
          "D": "D. conformal answer sets: empirical coverage (bar) vs nominal (tick) -- the guarantee holds"}
for k, (x0, y0, x1, y1) in reg.items():
    for li, line_ in enumerate(wrap_to(titles[k], 14, x1 - x0 - 24, bold=True)):
        T(x0 + 12, y0 + 8 + 17 * li, line_, 14, bold=True); fits(x0 + 12, y0 + 8 + 17 * li, line_, 14, [x0, y0, x1, y1], bold=True)
x0, y0, x1, y1 = reg["A"]; base = y1 - 40; slots = 6
for i, (p, kept, accv) in enumerate(DATA["both_ends"]):
    cx = x0 + 60 + (x1 - x0 - 80) * (i + 0.5) / slots
    T(cx, base + 6, "p>=%.1f" % p, 14, anchor="ma"); T(cx, base - max(kept, accv * 100) * (base - y0 - 70) / 100 - 18, "%.0f%% @ %.3f" % (kept, accv), 13, c=acc, anchor="ma")
for j, (nm, val) in enumerate(DATA["published"]):
    cx = x0 + 60 + (x1 - x0 - 80) * (4 + j + 0.5) / slots
    T(cx, base + 6, nm, 14, anchor="ma"); T(cx, base - val * (base - y0 - 70) / 100 - 18, "%.1f" % val, 13, c=dim, anchor="ma")
x0, y0, x1, y1 = reg["B"]; base = y1 - 40
for i, (nm, val) in enumerate(DATA["gate"]):
    cx = x0 + 60 + (x1 - x0 - 80) * (i + 0.5) / 4
    T(cx, base + 6, nm.replace("\n", " "), 14, anchor="ma"); T(cx, base - val * (base - y0 - 70) / 100 - 18, "%.1f%%" % val, 13, c=acc, anchor="ma")
x0, y0, x1, y1 = reg["C"]; base = y1 - 40
for i, (nm, fired, accv) in enumerate(DATA["reflex"]):
    cx = x0 + 60 + (x1 - x0 - 80) * (i + 0.5) / 2
    T(cx, base + 6, nm, 14, anchor="ma"); T(cx, base - fired * (base - y0 - 70) / 100 - 18, "fired %.1f%%  acc %.3f" % (fired, accv), 13, c=acc, anchor="ma")
x0, y0, x1, y1 = reg["D"]; base = y1 - 40
for i, (nm, cov) in enumerate(DATA["conformal"]):
    cx = x0 + 60 + (x1 - x0 - 80) * (i + 0.5) / 4
    T(cx, base + 6, nm, 14, anchor="ma"); T(cx, base - cov * (base - y0 - 70) - 18, "%.3f" % cov, 13, c=acc, anchor="ma")
T(60, 962, "substrate: NumPy only, deterministic, 5 ms/decision; every bar has a baseline and its seed spread in docs/research", 13, c=dim)
img.save(LABELS)
w4.post("/api/graph", {"nodes": [{"id": "labels", "type": "Media in", "params": {"source": LABELS, "play": 0}, "x": 0, "y": 0},
                                  {"id": "out", "type": "Output", "inputs": {"image": "labels"}, "x": 200, "y": 0}]})
w4.post("/api/graph/apply", {"id": "labels", "name": "labels"})
st = json.loads(w4.get("/api/state")); lab = [l["id"] for l in st["layers"] if l.get("name", "").startswith("labels")][-1]
w4.post("/api/layer", {"action": "edit", "id": lab, "blend": "add"})
w4.post("/api/graph", {"nodes": [{"id": "out", "type": "Output", "x": 0, "y": 0}]})
a = w4.analyze([{"name": "title", "box": [0.04, 0.03, 0.60, 0.08], "metrics": ["brightness"]}, {"name": "ground", "box": [0.01, 0.95, 0.03, 0.99], "metrics": ["brightness"]}])
ev = {r["name"]: r.get("brightness") for r in a.get("regions", [])}
ev["text_overflow"] = len(OVERFLOW)
if OVERFLOW:
    print("  OVERFLOW:", OVERFLOW[:4])
w4.step("bake the label overlay additively", "graph/apply", "title region brighter than the ground AND zero text boxes outside their panel", ev)
print("W4 labels:", {k: round(v, 3) for k, v in ev.items() if v is not None})

# ================================ 5. finish: export, verify, ledger ==========================================
open("/home/claude/bench_poster.png", "wb").write(w4.get("/api/composite.png"))
open("/home/claude/bench_poster.lews", "wb").write(w4.get("/api/workspace.lews"))
ev = core("swarm_evaluate", {"audits": ["catalog_gaps"]})
led = core("decision_records", {"k": 6})
print("evaluator all_ok:", ev["all_ok"], "| ledger:", led["stats"], "| %.0fs total" % (time.time() - t0))
