"""holographic_framebudget.py -- the FRAME-BUDGET CONTROLLER: one knob from target FPS to concrete render + sim
quality, held closed-loop against MEASURED frame time. The missing layer that ties the real-time pieces together.

THE PROBLEM
-----------
The engine already has the real-time PARTS -- render_adaptive (a resolution/sample knob), draft_vs_refine_simulation
(a sim grid/substep knob), the LOD chains, realtime_session (draft/refine) -- but nothing ties a TARGET FRAME RATE
to them. A front-end client that wants 60 fps needs someone to answer, every frame: "given I have 16.7 ms, what
resolution, how many samples, what sim grid?" and then to CORRECT when the last frame ran long. That is a control
loop, and it did not exist -- each part had its own quality dial and no conductor.

WHAT THIS IS
  * A QUALITY LADDER: an ordered list of presets, coarsest first, each naming a render resolution + sample count
    and a sim grid + substep count. Level 0 is the cheapest (always displayable); higher levels cost more.
  * A BUDGET: target_fps -> a per-frame millisecond budget (with a headroom fraction, because a frame that exactly
    fills the budget has already missed it -- vsync waits for the NEXT interval).
  * A CONTROLLER: after each frame you report the measured frame_ms; it drops a level when you blow the budget and
    climbs a level only when you are COMFORTABLY under it (hysteresis), so quality does not oscillate frame to frame.

THE HONEST CONSTRAINT (kept negative, from draft_vs_refine_simulation)
  For a RENDER, a coarse frame is a DRAFT of the fine one -- refining sharpens it, so dropping quality degrades
  gracefully toward the same image. For a CHAOTIC SIMULATION this is FALSE: fluid at grid 32 vs 48 has relative
  error 1.000 -- a coarse sim is a DIFFERENT trajectory, not a blurry version of the fine one. So the controller
  treats the sim quality honestly: dropping the sim level buys frame rate but CHANGES the simulation, it does not
  merely blur it. The ladder exposes render and sim levels SEPARATELY for exactly this reason -- you may want to
  hold the sim grid fixed (one trajectory) and trade only render quality, or accept a different-but-real-time solve.

NumPy / stdlib only. Deterministic: the controller's decisions are a pure function of the measured times.
"""

import numpy as np


# A default quality ladder, coarsest first. Each preset is a plain dict so a caller can swap in their own knobs;
# the controller only relies on the ORDER (index 0 cheapest) and reads whatever keys the caller's renderer/sim use.
DEFAULT_LADDER = [
    {"name": "potato",  "width": 128, "height": 128, "samples": 1, "sim_grid": 16, "sim_substeps": 1},
    {"name": "low",     "width": 192, "height": 192, "samples": 1, "sim_grid": 20, "sim_substeps": 1},
    {"name": "medium",  "width": 256, "height": 256, "samples": 2, "sim_grid": 24, "sim_substeps": 2},
    {"name": "high",    "width": 384, "height": 384, "samples": 3, "sim_grid": 32, "sim_substeps": 3},
    {"name": "ultra",   "width": 512, "height": 512, "samples": 4, "sim_grid": 48, "sim_substeps": 4},
]


def frame_budget_ms(target_fps, headroom=0.15):
    """Convert a target frame rate to a per-frame millisecond BUDGET, minus a headroom fraction. WHY headroom: a
    frame that exactly fills 1/fps seconds has already missed the vsync interval (present, swap, and the client's
    own work all need time), so the render+sim must finish INSIDE the interval. `headroom`=0.15 leaves 15% slack.
    E.g. 60 fps -> 16.67 ms nominal -> ~14.2 ms usable budget; 30 fps -> ~28.3 ms."""
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    return (1000.0 / target_fps) * (1.0 - headroom)


# Standard streaming canvas sizes -- the resolutions OBS scenes are almost always built at, so a browser source
# that matches one of these needs no scaling (scaling in OBS costs quality and CPU). Longest-edge keyed.
_STREAM_PRESETS = {
    "720p":  (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4k":    (3840, 2160),
}


def obs_capture_profile(base_url="http://127.0.0.1:5050/", preset="1080p", fps=30,
                        transparent=False, headroom=0.15):
    """Produce the settings a streamer types into OBS to capture this canvas as a BROWSER SOURCE -- the realistic,
    in-constitution way to put leOS on a stream. (A full RTMP/NDI/virtual-camera ENCODER is deliberately NOT here:
    it needs ffmpeg / OS video I/O, which the NumPy-only core forbids. OBS itself does the encoding once it is
    capturing the browser source, so the engine's job is only to serve a clean, correctly-sized page and tell the
    streamer how to point OBS at it.)

    base_url:    where the leOS web front end is served (the HTTP service root).
    preset:      one of '720p','1080p','1440p','4k' -- match your OBS canvas so OBS does no scaling.
    fps:         browser-source FPS. Kept modest by default (30) because the browser source re-renders the whole
                 page each frame; 60 is fine for a fast machine.
    transparent: if True, advertise a transparent background so OBS composites the canvas over other sources
                 (the front end must actually render with an alpha/transparent clear for this to take effect --
                 this profile only carries the flag + the OBS-side CSS, it cannot force the page to be transparent).
    headroom:    passed to frame_budget_ms -- the per-frame ms budget the renderer must finish inside to hold `fps`.

    Returns a plain dict (JSON-friendly, so it round-trips over /invoke) with:
      url                 the browser-source URL (base_url; a fragment hint is added when transparent)
      width, height       the browser-source size in px (from the preset)
      fps                 the browser-source FPS to enter
      frame_budget_ms     the wall-clock budget per frame at that fps (reuses frame_budget_ms)
      transparent         echoed flag
      custom_css          the OBS 'Custom CSS' to paste (empty background when transparent; else '')
      obs_steps           an ordered list of human steps to add the source
      note                the honest boundary (browser source vs a native encoder)

    Deterministic, stdlib-only. This is guidance + numbers, not a live pipe; the live frames come from the existing
    /frame and /frame/stream endpoints, which OBS's browser source consumes by simply rendering the page."""
    if preset not in _STREAM_PRESETS:
        raise ValueError("preset must be one of %s, got %r" % (sorted(_STREAM_PRESETS), preset))
    if fps <= 0:
        raise ValueError("fps must be positive")
    w, h = _STREAM_PRESETS[preset]
    # OBS's browser source honours a transparent page background; the canonical CSS is to clear the body so the
    # page's own alpha shows through. When not transparent we leave custom CSS empty (OBS default is fine).
    css = "body { background: rgba(0, 0, 0, 0); margin: 0; overflow: hidden; }" if transparent else ""
    url = base_url
    if transparent and "#" not in url:
        url = url + "#transparent"        # a hint the front end can read to pick a transparent clear colour
    steps = [
        "In OBS, add a Source -> Browser.",
        "Set URL to %s" % url,
        "Set Width to %d and Height to %d (match your OBS canvas so there is no scaling)." % (w, h),
        "Set FPS to %d." % fps,
    ]
    if transparent:
        steps.append("Paste the custom_css below into the 'Custom CSS' box so the background is transparent, "
                     "and make sure the leOS canvas is rendering with a transparent clear.")
    steps.append("Click OK; the canvas appears as a source you can resize/crop like any other.")
    return {
        "url": url,
        "width": w,
        "height": h,
        "fps": int(fps),
        "frame_budget_ms": round(frame_budget_ms(fps, headroom=headroom), 3),
        "transparent": bool(transparent),
        "custom_css": css,
        "obs_steps": steps,
        "note": "Browser-source capture: OBS renders this page and does the video encoding. leOS serves the page "
                "and (via /frame and /frame/stream) the frames; it does not itself encode RTMP/NDI/a virtual "
                "camera (that needs ffmpeg/OS video I/O, outside the NumPy-only core).",
    }


class FrameBudgetController:
    """A closed-loop quality controller: hold a target FPS by moving up and down a quality ladder based on MEASURED
    frame time. Usage each frame:

        ctrl = FrameBudgetController(target_fps=60, ladder=DEFAULT_LADDER)
        preset = ctrl.current()                 # the settings to render/simulate this frame with
        ... render + simulate, measure elapsed ...
        ctrl.report(frame_ms)                   # feed back the measured time; the level adapts for NEXT frame

    The controller drops a level the moment a frame blows the budget (react fast to a stall), but climbs a level
    only after several frames sit COMFORTABLY under it (climb slow, to avoid oscillating on the edge). This
    asymmetry is deliberate: a dropped frame is visible, a slightly-too-low quality is not.
    """

    def __init__(self, target_fps=60, ladder=None, headroom=0.15, start_level=None,
                 climb_after=8, climb_margin=0.6):
        self.ladder = list(ladder) if ladder is not None else list(DEFAULT_LADDER)
        if len(self.ladder) == 0:
            raise ValueError("ladder must have at least one preset")
        self.target_fps = float(target_fps)
        self.budget_ms = frame_budget_ms(target_fps, headroom=headroom)
        # start in the MIDDLE by default (a reasonable guess; the loop finds the right level within a few frames).
        self.level = int(start_level) if start_level is not None else len(self.ladder) // 2
        self.level = max(0, min(len(self.ladder) - 1, self.level))
        self.climb_after = int(climb_after)       # consecutive comfortable frames before climbing
        self.climb_margin = float(climb_margin)   # 'comfortable' = frame_ms < climb_margin * budget
        self._good_streak = 0
        self.history = []                         # (frame_ms, level, action) per report, for measurement
        self._blocked_above = len(self.ladder)    # do not climb into a level known to miss (set on a drop);
        #                                           reset when the budget clearly has new headroom (see report).

    def current(self):
        """The preset (a dict of quality knobs) to render/simulate the NEXT frame with -- the current ladder rung."""
        return dict(self.ladder[self.level])

    def report(self, frame_ms):
        """Feed back the MEASURED time (ms) the last frame took. Returns the action taken: 'drop', 'climb', or
        'hold'. Drops immediately on a budget miss; climbs only after `climb_after` comfortable frames. Updates the
        current level for the next `current()` call."""
        frame_ms = float(frame_ms)
        action = "hold"
        if frame_ms > self.budget_ms and self.level > 0:
            # a miss -- react immediately, one rung down, and reset the climb streak. Remember the level we just
            # fell FROM as 'blocked': re-probing it wastes a frame every climb cycle (the classic boundary churn),
            # so we won't climb back into it until clear new headroom appears (a very cheap frame, below).
            self._blocked_above = self.level
            self.level -= 1
            self._good_streak = 0
            action = "drop"
        elif frame_ms < self.climb_margin * self.budget_ms:
            # comfortably under -- count toward a climb. A frame this cheap is evidence of headroom, so if we are
            # well under (< half the climb margin) allow re-probing a previously-blocked level: the content may
            # have lightened. Otherwise respect the block to avoid futile oscillation.
            if frame_ms < 0.5 * self.climb_margin * self.budget_ms:
                self._blocked_above = len(self.ladder)
            self._good_streak += 1
            if (self._good_streak >= self.climb_after and self.level < len(self.ladder) - 1
                    and self.level + 1 < self._blocked_above):
                self.level += 1
                self._good_streak = 0
                action = "climb"
        else:
            # in the band [climb_margin*budget, budget] -- right where we want to be; hold and reset the streak so a
            # brief dip below climb_margin doesn't accumulate toward an unwanted climb.
            self._good_streak = 0
        self.history.append((frame_ms, self.level, action))
        return action

    def stats(self):
        """A measured summary of the session so far: the fraction of reported frames that MET the budget (the
        thing the client cares about), the mean frame time, and how often the level changed (churn -- high churn
        means the ladder steps are too coarse or the content too spiky). Returns a dict."""
        if not self.history:
            return {"frames": 0, "met_budget_frac": None, "mean_frame_ms": None, "changes": 0}
        times = np.array([h[0] for h in self.history])
        met = float(np.mean(times <= self.budget_ms))
        changes = sum(1 for h in self.history if h[2] != "hold")
        return {"frames": len(self.history), "met_budget_frac": round(met, 4),
                "mean_frame_ms": round(float(times.mean()), 3), "budget_ms": round(self.budget_ms, 3),
                "final_level": self.level, "changes": changes}


class FrameServer:
    """Server-side frame serving for a front-end client that pulls frames (the request/response form of frame
    streaming -- no websocket needed, which this stdlib service doesn't have). Holds one FrameBudgetController per
    client SESSION and answers a per-frame request: given the session, its target fps, and how long its LAST frame
    took, return the quality preset to use for the NEXT frame plus the running stats. The client renders/simulates
    at that quality and reports back the next time -- a closed loop across the network.

    WHY a per-session controller (not one global): two clients on the same node may want different frame rates (a
    phone at 30, a desktop at 60) and have different hardware, so each needs its own budget and its own adaptation
    state. The session id keys them apart; an unknown session is created on first contact with its stated fps.

    This is deliberately STATELESS about the pixels -- it decides QUALITY, not content. The content comes from the
    mind's render/sim faculties (realtime_session.payload(), render_adaptive, draft_vs_refine_simulation), which
    the caller invokes with the returned preset. Keeping the two separate means the frame server works for any
    renderer, and the quality logic stays testable without a GPU."""

    def __init__(self, ladder=None, headroom=0.15):
        self._sessions = {}                                   # session id -> FrameBudgetController
        self._ladder = ladder
        self._headroom = float(headroom)

    def next_frame(self, session, target_fps=60, last_frame_ms=None):
        """The per-frame decision. `session` is a client id; `target_fps` its desired rate (used only when the
        session is first seen); `last_frame_ms` is how long the client's PREVIOUS frame took (None on the first
        request). Returns a dict: `preset` (the quality knobs to use now), `level` (its ladder index), `budget_ms`,
        and `stats` (met-budget fraction so far). Creates the session's controller on first contact."""
        ctrl = self._sessions.get(session)
        if ctrl is None:
            ctrl = FrameBudgetController(target_fps=target_fps, ladder=self._ladder, headroom=self._headroom)
            self._sessions[session] = ctrl
        if last_frame_ms is not None:
            ctrl.report(float(last_frame_ms))                 # close the loop with the reported time
        preset = ctrl.current()
        return {"session": str(session), "preset": preset, "level": ctrl.level,
                "budget_ms": round(ctrl.budget_ms, 3), "target_fps": ctrl.target_fps, "stats": ctrl.stats()}

    def drop_session(self, session):
        """Forget a session's controller (the client disconnected). Returns True if it existed."""
        return self._sessions.pop(session, None) is not None

    def serve_frame(self, session, render_fn, target_fps=60):
        """The ONE-CALL frame loop: pick the quality, render at it, TIME the render, and close the loop -- so the
        caller never has to measure or report the frame time by hand. `render_fn(preset) -> payload` produces the
        frame content at the given quality preset (pixels/mesh/shader/whatever, JSON-safe). Returns {session,
        preset, level, budget_ms, target_fps, frame_ms (the MEASURED render time), payload, stats}. The measured
        time is fed back automatically, so calling serve_frame in a loop holds the target fps with adaptive quality
        AND returns the content inline -- one round-trip per frame instead of three (decide quality, render, fetch).

        WHY time it here: the honest frame time is the time THIS render actually took, not a number the client
        self-reports (which it may fudge or forget). Measuring at the point of render keeps the loop truthful."""
        import time
        ctrl = self._sessions.get(session)
        if ctrl is None:
            ctrl = FrameBudgetController(target_fps=target_fps, ladder=self._ladder, headroom=self._headroom)
            self._sessions[session] = ctrl
        preset = ctrl.current()
        t0 = time.perf_counter()
        payload = render_fn(preset)                           # produce the content at the chosen quality
        frame_ms = (time.perf_counter() - t0) * 1000.0
        ctrl.report(frame_ms)                                 # close the loop with the MEASURED time
        return {"session": str(session), "preset": preset, "level": ctrl.level,
                "budget_ms": round(ctrl.budget_ms, 3), "target_fps": ctrl.target_fps,
                "frame_ms": round(frame_ms, 3), "payload": payload, "stats": ctrl.stats()}

    def serve_frame_distributed(self, session, distribute_bricks, target_fps=60, tiles=(2, 2), t=0.0):
        """Serve one frame whose PIXELS are rendered by TILES distributed across workers -- the real-time pipeline
        running on the distributed-compute machinery (holographic_distribute.distribute_bricks). `distribute_bricks`
        is the mind's faculty (pass `mind.distribute_bricks`); it runs a worker per image region and places the
        results seamlessly, so the frame is assembled from `tiles`=(rows, cols) disjoint bricks. Picks the quality,
        renders the demo SDF per tile (each tile raymarches only its own pixels), TIMES the whole distributed
        render, and closes the loop -- so distributing the work still holds the target fps.

        WHY this matters: a single node caps the resolution it can hit in a frame budget; splitting the frame across
        workers lifts that cap while the SAME frame-budget controller keeps the rate honest. Tiles are disjoint and
        order-independent (distribute_bricks guarantees seamless placement), so the distributed frame is bit-
        identical to a single-node render of the same preset. Returns the same shape as serve_frame plus
        `tiles_ran`. The quality knob (preset resolution) and the parallelism knob (tiles) are independent."""
        import time
        ctrl = self._sessions.get(session)
        if ctrl is None:
            ctrl = FrameBudgetController(target_fps=target_fps, ladder=self._ladder, headroom=self._headroom)
            self._sessions[session] = ctrl
        preset = ctrl.current()
        w = int(preset.get("width", 128)); h = int(preset.get("height", 128))
        nr, nc = int(tiles[0]), int(tiles[1])
        # build disjoint row/col brick regions covering the whole image.
        rbounds = [round(i * h / nr) for i in range(nr + 1)]
        cbounds = [round(j * w / nc) for j in range(nc + 1)]
        regions = [(slice(rbounds[i], rbounds[i + 1]), slice(cbounds[j], cbounds[j + 1]))
                   for i in range(nr) for j in range(nc)]

        def tile_worker(region, _cache):
            # render ONLY this tile's pixels: a sub-preset with the tile's width/height and a pixel offset so the
            # rays match the full-frame camera (a tile is a window into the same image, not a separate render).
            rs, cs = region
            return _demo_pixels_region(preset, t, rs, cs, h, w)

        t0 = time.perf_counter()
        out, info = distribute_bricks((h, w), regions, tile_worker)
        frame_ms = (time.perf_counter() - t0) * 1000.0
        ctrl.report(frame_ms)
        payload = {"width": w, "height": h, "pixels": np.asarray(out, np.uint8).tolist(), "kinds": ["pixels"]}
        return {"session": str(session), "preset": preset, "level": ctrl.level,
                "budget_ms": round(ctrl.budget_ms, 3), "target_fps": ctrl.target_fps,
                "frame_ms": round(frame_ms, 3), "payload": payload,
                "tiles_ran": info.get("ran", len(regions)), "stats": ctrl.stats()}

    def sessions(self):
        """The active session ids and their current level -- for a health/status view."""
        return {s: c.level for s, c in self._sessions.items()}


PROJECTION_KINDS = ("pixels", "mesh", "splats", "shader", "lod", "glsl", "wgsl", "ascii", "wireframe")


def _demo_sdf(px, py, pz, sphere_y):
    """The demo scene's signed-distance field, as a pure function: a unit sphere bobbing at height `sphere_y`
    unioned with a ground plane at y=-1. One source of truth so every projection (pixels/mesh/splats/lod) is of the
    SAME field -- the 'one scene, many outputs, no drift' contract, in miniature."""
    sd = np.sqrt(px * px + (py - sphere_y) ** 2 + pz * pz) - 1.0
    gd = py + 1.0
    return np.minimum(sd, gd)


def _demo_pixels(preset, t):
    """Raymarch the demo SDF to grayscale pixel rows at the preset resolution (cost scales with resolution+samples)."""
    w = int(preset.get("width", 128)); h = int(preset.get("height", 128))
    samples = int(preset.get("samples", 1))
    ys, xs = np.mgrid[0:h, 0:w]
    u = (xs / max(w - 1, 1)) * 2.0 - 1.0
    v = (ys / max(h - 1, 1)) * 2.0 - 1.0
    u *= w / max(h, 1)
    dx, dy, dz = u, -v, -np.ones_like(u) * 1.6
    norm = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = dx / norm, dy / norm, dz / norm
    sphere_y = 0.3 * np.sin(t)
    px, py, pz = np.zeros_like(u), np.zeros_like(u), np.full_like(u, 3.0)
    hit = np.zeros((h, w), bool); depth = np.zeros((h, w), float)
    for _ in range(12 + 6 * samples):
        d = _demo_sdf(px, py, pz, sphere_y)
        step = np.where(hit, 0.0, np.maximum(d, 1e-3))
        px += dx * step; py += dy * step; pz += dz * step
        depth += step
        hit |= (d < 1e-2)
    shade = np.where(hit, np.clip(1.4 - 0.25 * depth, 0.0, 1.0), 0.05)
    return {"width": w, "height": h, "pixels": (shade * 255).astype(np.uint8).tolist()}


def _demo_surface_points(preset, t, n):
    """Sample n points ON the demo sphere's surface (deterministic Fibonacci sphere), for mesh/splats projections."""
    sphere_y = 0.3 * np.sin(t)
    i = np.arange(n)
    phi = np.arccos(1.0 - 2.0 * (i + 0.5) / n)                # even polar spread
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i                    # golden angle
    x = np.sin(phi) * np.cos(theta)
    y = np.cos(phi) + sphere_y
    z = np.sin(phi) * np.sin(theta)
    return np.stack([x, y, z], axis=1)


def _demo_wireframe(preset, t):
    """Build a WIREFRAME CAGE of the demo sphere -- the editable overlay a 3D-modeling app shows so a user can grab
    verts, edges, and faces. Returns {vertices, edges, faces, normals} where `edges` is a list of [i,j] vertex-index
    pairs (each quad's boundary, DEDUPED so a shared edge appears once) and `faces` is a list of vertex-index quads.
    This is the interaction surface: the app raycasts against these verts/edges/faces to pick and edit, and can
    toggle its visibility as an overlay over pixels/ascii. A quad-sphere so the cage looks like the grid a modeler
    expects, not a triangle soup."""
    nu = int(preset.get("cage_u", 12))                        # longitude divisions
    nv = int(preset.get("cage_v", 8))                         # latitude divisions
    sphere_y = 0.3 * np.sin(float(t))
    verts = []
    for iv in range(nv + 1):
        lat = np.pi * iv / nv                                 # 0..pi
        for iu in range(nu):
            lon = 2 * np.pi * iu / nu
            verts.append([float(np.sin(lat) * np.cos(lon)),
                          float(np.cos(lat) + sphere_y),
                          float(np.sin(lat) * np.sin(lon))])

    def vid(iu, iv):
        return iv * nu + (iu % nu)                            # wrap longitude

    faces, edge_set = [], set()
    for iv in range(nv):
        for iu in range(nu):
            a, b, c, d = vid(iu, iv), vid(iu + 1, iv), vid(iu + 1, iv + 1), vid(iu, iv + 1)
            faces.append([a, b, c, d])
            for (p, q) in ((a, b), (b, c), (c, d), (d, a)):
                edge_set.add((min(p, q), max(p, q)))          # dedupe shared edges
    edges = [list(e) for e in sorted(edge_set)]
    normals = [[v[0], v[1] - sphere_y, v[2]] for v in verts]  # outward unit normals for lit cage handles
    return {"vertices": verts, "edges": edges, "faces": faces, "normals": normals}


def _demo_ascii(preset, t):
    """A braille/ascii raymarch of the demo SDF -- the 'see it over SSH' output, and a base layer a wireframe
    overlay can toggle on top of. Width scales with the preset so cost tracks quality. Returns {ascii, width}."""
    w = max(24, int(preset.get("width", 128)) // 4)           # ascii is coarse; quarter the pixel width
    h = max(12, w // 2)
    sphere_y = 0.3 * np.sin(float(t))
    ramp = " .:-=+*#%@"
    rows = []
    for j in range(h):
        line = []
        for i in range(w):
            u = (i / (w - 1)) * 2.0 - 1.0
            v = (j / (h - 1)) * 2.0 - 1.0
            n = (u * u + v * v + 1.6 * 1.6) ** 0.5
            dx, dy, dz = u / n, -v / n, -1.6 / n
            px, py, pz, depth, hit = 0.0, 0.0, 3.0, 0.0, False
            for _ in range(24):
                d = float(_demo_sdf(np.array(px), np.array(py), np.array(pz), sphere_y))
                if d < 1e-2:
                    hit = True; break
                px += dx * d; py += dy * d; pz += dz * d; depth += d
                if depth > 8:
                    break
            line.append(ramp[min(len(ramp) - 1, max(0, int((1.4 - 0.2 * depth) * (len(ramp) - 1))))] if hit else " ")
        rows.append("".join(line))
    return {"ascii": "\n".join(rows), "width": w}


def demo_frame_payload(preset, t=0.0, kinds=("pixels",)):
    """A self-contained frame renderer for the HTTP /frame endpoint that supports ALL projection types -- the
    caller picks which output(s) to receive, and MULTIPLE simultaneously. `kinds` is any subset of
    ('pixels','mesh','splats','shader','lod'); every requested projection is of the SAME demo SDF (a bobbing sphere
    + ground plane), so the outputs cannot drift -- the 'one scene, many outputs' contract in miniature:

      * pixels -- a raymarched grayscale image (width,height,pixels rows), the default.
      * mesh   -- {vertices, faces} sampling the sphere surface (a triangle fan over the point cloud).
      * splats -- a list of {position, scale} billboard proxies on the surface (the browser point-cloud format).
      * shader -- a WGSL source string for the SDF, so the browser GPU runs a projection of the same field.
      * lod    -- a progressive descriptor of the field sampled on a small grid (per-level bytes + rel_rms).

    Returns a dict keyed by the requested kinds (plus 'width'/'height' when pixels is present). Deterministic given
    (preset, t, kinds). NumPy/stdlib only. A production client renders its OWN scene via serve_frame's callback;
    this is the built-in fallback so POST /frame returns displayable content in any format out of the box."""
    kinds = tuple(kinds) if not isinstance(kinds, str) else (kinds,)
    bad = [k for k in kinds if k not in PROJECTION_KINDS]
    if bad:
        raise ValueError("unknown projection kind(s) %r; known: %s" % (bad, list(PROJECTION_KINDS)))
    t = float(t)
    sphere_y = 0.3 * np.sin(t)
    out = {"kinds": list(kinds), "t": t}
    for kind in kinds:
        if kind == "pixels":
            out.update(_demo_pixels(preset, t))                # merges width/height/pixels
        elif kind == "mesh":
            pts = _demo_surface_points(preset, t, n=int(preset.get("mesh_points", 200)))
            # a simple fan triangulation over consecutive points -- enough for a real, displayable mesh payload.
            faces = [[0, i, i + 1] for i in range(1, len(pts) - 1)]
            out["mesh"] = {"vertices": pts.tolist(), "faces": faces}
        elif kind == "splats":
            pts = _demo_surface_points(preset, t, n=int(preset.get("n_splats", 120)))
            scale = 0.06 + 0.02 * preset.get("samples", 1)
            out["splats"] = [{"position": p.tolist(), "scale": round(float(scale), 4)} for p in pts]
        elif kind in ("shader", "wgsl"):
            out[kind] = {"language": "wgsl", "source":
                         "fn map(p: vec3<f32>) -> f32 {\n"
                         "  let s = length(vec3<f32>(p.x, p.y - %.4f, p.z)) - 1.0;\n"
                         "  let g = p.y + 1.0;\n  return min(s, g);\n}" % sphere_y}
        elif kind == "glsl":
            # GLSL (Shadertoy-style) -- the best fit for a web VIEWPORT, per the modeling-app use case.
            out["glsl"] = {"language": "glsl", "source":
                           "float map(vec3 p) {\n"
                           "  float s = length(vec3(p.x, p.y - %.4f, p.z)) - 1.0;\n"
                           "  float g = p.y + 1.0;\n  return min(s, g);\n}" % sphere_y}
        elif kind == "ascii":
            out["ascii"] = _demo_ascii(preset, t)
        elif kind == "wireframe":
            out["wireframe"] = _demo_wireframe(preset, t)
        elif kind == "lod":
            from holographic.io_and_interop.holographic_stream import stream_encode
            g = np.linspace(-2.0, 2.0, 12)
            GX, GY, GZ = np.meshgrid(g, g, g, indexing="ij")
            field = _demo_sdf(GX, GY, GZ, sphere_y)
            out["lod"] = stream_encode(field)["descriptor"]
    return out


def _demo_pixels_region(preset, t, rs, cs, full_h, full_w):
    """Render ONLY the pixel rows `rs` x cols `cs` of the demo frame, as a window into the SAME full-frame camera
    (the ray for pixel (y,x) is identical whether rendered alone or in a tile). This is what makes distributed
    tiling seamless and bit-identical to a single-node render: every tile computes a disjoint slice of the one
    image. Returns the tile as a (tile_h, tile_w) uint8 array."""
    samples = int(preset.get("samples", 1))
    ys, xs = np.mgrid[rs.start:rs.stop, cs.start:cs.stop]     # ABSOLUTE pixel coords, so rays match the full frame
    u = (xs / max(full_w - 1, 1)) * 2.0 - 1.0
    v = (ys / max(full_h - 1, 1)) * 2.0 - 1.0
    u = u * (full_w / max(full_h, 1))
    dx, dy, dz = u, -v, -np.ones_like(u) * 1.6
    norm = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = dx / norm, dy / norm, dz / norm
    sphere_y = 0.3 * np.sin(float(t))
    px, py, pz = np.zeros_like(u), np.zeros_like(u), np.full_like(u, 3.0)
    hit = np.zeros(u.shape, bool); depth = np.zeros(u.shape, float)
    for _ in range(12 + 6 * samples):
        d = _demo_sdf(px, py, pz, sphere_y)
        step = np.where(hit, 0.0, np.maximum(d, 1e-3))
        px += dx * step; py += dy * step; pz += dz * step
        depth += step
        hit |= (d < 1e-2)
    shade = np.where(hit, np.clip(1.4 - 0.25 * depth, 0.0, 1.0), 0.05)
    return (shade * 255).astype(np.uint8)


def pick_element(wireframe, screen_u, screen_v, want="vertex", cam_z=3.0, fov_scale=1.6):
    """VIEWPORT PICKING for a 3D-modeling app: given a wireframe cage and a screen coordinate (`screen_u`,
    `screen_v` in -1..1, the normalized viewport position under the cursor), return which element the user is
    pointing at. `want` is 'vertex', 'edge', or 'face'. Casts a ray from the same camera the demo renders with
    (looking down -z from cam_z) and returns {kind, index, distance, position} for the nearest matching element --
    the pick a modeling app needs to select and then edit a vert/edge/face. Returns {kind, index:None} on a miss
    (the ray hit nothing near an element).

    This is deliberately geometry-only (no GPU pick buffer): it projects the cage's own verts to the screen and
    finds the closest, which is exact, deterministic, and needs no render pass -- the app already has the cage."""
    verts = np.asarray(wireframe["vertices"], float)
    if len(verts) == 0:
        return {"kind": want, "index": None}
    # project each vertex to the screen with the demo camera (perspective divide), then measure screen distance.
    rel = verts - np.array([0.0, 0.0, cam_z])                 # camera at (0,0,cam_z) looking -z
    z = -rel[:, 2]                                            # depth in front of the camera (>0 visible)
    visible = z > 1e-3
    su = np.where(visible, rel[:, 0] / (z * (1.0 / fov_scale) + 1e-9), 1e9)
    sv = np.where(visible, -rel[:, 1] / (z * (1.0 / fov_scale) + 1e-9), 1e9)
    d2 = (su - screen_u) ** 2 + (sv - screen_v) ** 2          # squared screen distance to each vertex

    if want == "vertex":
        j = int(np.argmin(d2))
        if not visible[j]:
            return {"kind": "vertex", "index": None}
        return {"kind": "vertex", "index": j, "distance": float(np.sqrt(d2[j])),
                "position": verts[j].tolist()}

    if want == "edge":
        best, best_d = None, 1e18
        for ei, (a, b) in enumerate(wireframe["edges"]):
            if not (visible[a] and visible[b]):
                continue
            # distance from the cursor to the projected edge segment (a..b) in screen space.
            pa = np.array([su[a], sv[a]]); pb = np.array([su[b], sv[b]])
            ab = pb - pa; L2 = float(ab @ ab) + 1e-12
            tt = np.clip(((np.array([screen_u, screen_v]) - pa) @ ab) / L2, 0.0, 1.0)
            proj = pa + tt * ab
            dd = float((proj - np.array([screen_u, screen_v])) @ (proj - np.array([screen_u, screen_v])))
            if dd < best_d:
                best_d, best = dd, ei
        if best is None:
            return {"kind": "edge", "index": None}
        return {"kind": "edge", "index": best, "distance": float(np.sqrt(best_d)),
                "vertices": list(wireframe["edges"][best])}

    if want == "face":
        best, best_d = None, 1e18
        for fi, face in enumerate(wireframe["faces"]):
            if not all(visible[v] for v in face):
                continue
            cu = float(np.mean([su[v] for v in face])); cv = float(np.mean([sv[v] for v in face]))
            dd = (cu - screen_u) ** 2 + (cv - screen_v) ** 2  # nearest face by projected centroid
            if dd < best_d:
                best_d, best = dd, fi
        if best is None:
            return {"kind": "face", "index": None}
        return {"kind": "face", "index": best, "distance": float(np.sqrt(best_d)),
                "vertices": list(wireframe["faces"][best])}

    raise ValueError("want must be 'vertex', 'edge', or 'face'; got %r" % want)


def _selftest():
    """Contracts:

    1. frame_budget_ms: 60 fps gives a ~14 ms usable budget, 30 fps ~28 ms; higher fps -> tighter budget.
    2. The controller DROPS quality when frames blow the budget and settles at a sustainable level (does not keep
       dropping past a level that fits).
    3. The controller CLIMBS quality when frames are comfortably cheap, but only after a streak (hysteresis -- one
       cheap frame does not trigger a climb).
    4. Closed loop on a SIMULATED cost model (cost grows with the ladder level): starting anywhere, the controller
       converges to the highest level that fits the budget and MEETS the budget on most frames.
    5. Determinism: identical reported times -> identical decisions.
    """
    # (1) budget math.
    b60 = frame_budget_ms(60)
    b30 = frame_budget_ms(30)
    assert 13.0 < b60 < 15.0 and 27.0 < b30 < 29.0, (b60, b30)
    assert b60 < b30                                            # more fps -> tighter budget

    # (2) drop on a miss.
    ctrl = FrameBudgetController(target_fps=60, start_level=4)  # start at ultra
    a = ctrl.report(40.0)                                      # way over the ~14 ms budget
    assert a == "drop" and ctrl.level == 3

    # (3) climb only after a streak.
    ctrl2 = FrameBudgetController(target_fps=60, start_level=0, climb_after=8, climb_margin=0.6)
    acts = [ctrl2.report(1.0) for _ in range(7)]               # 7 cheap frames -- not yet enough
    assert all(x == "hold" for x in acts) and ctrl2.level == 0
    assert ctrl2.report(1.0) == "climb" and ctrl2.level == 1   # the 8th triggers the climb

    # (4) closed loop against a cost model: frame_ms rises with the level. Pick a model where level 2 is the
    #     highest that fits the 60 fps budget, and confirm the controller finds and holds it.
    def cost_ms(level):
        return [4.0, 8.0, 12.0, 22.0, 40.0][level]            # levels 0..2 fit (<14.2), 3..4 blow it
    ctrl3 = FrameBudgetController(target_fps=60, start_level=4, climb_after=5)
    for _ in range(80):
        lvl = ctrl3.level
        ctrl3.report(cost_ms(lvl))
    st = ctrl3.stats()
    assert ctrl3.level == 2, ("should settle at the highest fitting level", ctrl3.level)
    assert st["met_budget_frac"] > 0.85, st                    # meets the budget on the vast majority of frames

    # (5) determinism.
    def run():
        c = FrameBudgetController(target_fps=60, start_level=4, climb_after=5)
        for _ in range(50):
            c.report(cost_ms(c.level))
        return [h[1] for h in c.history]
    assert run() == run()

    # (6) FrameServer: per-session controllers, closed loop across the (simulated) network. Two sessions at
    #     different fps get different budgets; reporting a slow frame drops the returned preset's level.
    fs = FrameServer()
    r0 = fs.next_frame("clientA", target_fps=60)              # first contact -- no last_frame_ms
    assert r0["preset"]["name"] == fs._sessions["clientA"].current()["name"]
    lvl0 = r0["level"]
    r1 = fs.next_frame("clientA", last_frame_ms=40.0)         # a slow frame -> drop
    assert r1["level"] == lvl0 - 1, (lvl0, r1["level"])
    # a second session at 30 fps has a looser budget than the 60 fps one.
    rB = fs.next_frame("clientB", target_fps=30)
    assert rB["budget_ms"] > r0["budget_ms"]
    assert set(fs.sessions().keys()) == {"clientA", "clientB"}
    assert fs.drop_session("clientA") and not fs.drop_session("nope")

    # (7) serve_frame: the one-call loop times the render itself and returns the payload INLINE. A render_fn whose
    #     cost we control drives the controller down when it's slow.
    fs2 = FrameServer()
    calls = {"n": 0}
    def slow_render(preset):
        calls["n"] += 1
        # simulate work proportional to resolution so the timing is real (a tiny numpy load).
        _ = np.sin(np.random.default_rng(0).random((preset["height"], preset["width"]))).sum()
        return {"drew": preset["name"]}
    out = fs2.serve_frame("c", slow_render, target_fps=60)
    assert out["payload"]["drew"] == out["preset"]["name"]     # payload came back inline
    assert "frame_ms" in out and out["frame_ms"] >= 0.0        # it measured the render itself
    assert calls["n"] == 1                                     # rendered exactly once

    # (8) demo_frame_payload: all projection kinds, selectable and simultaneous, each of the SAME field.
    pot = demo_frame_payload(DEFAULT_LADDER[0], kinds=("pixels",))       # potato 128x128 pixels
    ult = demo_frame_payload(DEFAULT_LADDER[-1], kinds=("pixels",))      # ultra 512x512
    assert pot["width"] == 128 and len(pot["pixels"]) == 128 and len(pot["pixels"][0]) == 128
    assert ult["width"] == 512 and len(ult["pixels"]) == 512
    assert all(0 <= px <= 255 for px in pot["pixels"][64])              # valid grayscale
    assert demo_frame_payload(DEFAULT_LADDER[0]) == demo_frame_payload(DEFAULT_LADDER[0])  # deterministic
    assert demo_frame_payload(DEFAULT_LADDER[0], t=0.0) != demo_frame_payload(DEFAULT_LADDER[0], t=1.0)  # animates

    # every projection kind is producible, and MULTIPLE at once.
    for k in PROJECTION_KINDS:
        one = demo_frame_payload(DEFAULT_LADDER[1], kinds=(k,))
        assert k in one, ("missing projection", k)
    multi = demo_frame_payload(DEFAULT_LADDER[1], kinds=("pixels", "mesh", "splats", "shader", "lod"))
    assert all(k in multi for k in ("pixels", "mesh", "splats", "shader", "lod")), multi["kinds"]  # all five
    assert "vertices" in multi["mesh"] and "faces" in multi["mesh"]
    assert isinstance(multi["splats"], list) and "position" in multi["splats"][0]
    assert multi["shader"]["language"] == "wgsl" and "map(" in multi["shader"]["source"]
    assert isinstance(multi["lod"], (dict, list))

    # (8b) the 3D-modeling-app outputs: glsl (web viewport), ascii (SSH), wireframe cage (editable), all of the
    #      SAME scene. Plus viewport PICKING of vert/edge/face.
    modeling = demo_frame_payload(DEFAULT_LADDER[1], kinds=("glsl", "ascii", "wireframe"))
    assert modeling["glsl"]["language"] == "glsl" and "map(" in modeling["glsl"]["source"]
    assert "ascii" in modeling["ascii"] and modeling["ascii"]["ascii"].count("\n") > 5
    wf = modeling["wireframe"]
    assert len(wf["vertices"]) > 0 and len(wf["edges"]) > 0 and len(wf["faces"]) > 0
    assert len(wf["normals"]) == len(wf["vertices"])
    # edges are deduped index pairs; faces are index tuples into vertices.
    assert all(0 <= i < len(wf["vertices"]) and 0 <= j < len(wf["vertices"]) for i, j in wf["edges"])
    # picking: the vertex nearest screen-centre is the front pole (0,0,~1), distance ~0.
    pv = pick_element(wf, 0.0, 0.0, want="vertex")
    assert pv["index"] is not None and abs(pv["position"][2] - 1.0) < 0.3 and pv["distance"] < 0.1
    pe = pick_element(wf, 0.2, -0.1, want="edge")
    assert pe["index"] is not None and len(pe["vertices"]) == 2
    pf = pick_element(wf, 0.2, -0.1, want="face")
    assert pf["index"] is not None and len(pf["vertices"]) >= 3
    assert pick_element(wf, 0.0, 0.0, want="vertex")["index"] == pv["index"]   # deterministic
    # an unknown kind is rejected loudly (not silently dropped).
    try:
        demo_frame_payload(DEFAULT_LADDER[0], kinds=("hologram",))
        assert False, "unknown kind must raise"
    except ValueError:
        pass

    # (9) DISTRIBUTED tiled render: serving a frame via tiles across workers is BIT-IDENTICAL to a single-node
    #     render of the same preset (the tiles are disjoint windows into the same image). Uses a local
    #     distribute_bricks stand-in so the module self-tests without the full mind.
    def local_distribute(out_shape, regions, worker, cache=None):
        out = np.zeros(out_shape, np.uint8)
        for reg in regions:
            out[reg] = worker(reg, None)
        return out, {"ran": len(regions), "skipped": 0}
    fs3 = FrameServer()
    dist = fs3.serve_frame_distributed("d", local_distribute, target_fps=30, tiles=(2, 2), t=0.5)
    assert dist["payload"]["pixels"] == _demo_pixels(dist["preset"], 0.5)["pixels"], "tiled == single-node render"
    assert dist["tiles_ran"] == 4 and "frame_ms" in dist

    # (10) obs_capture_profile: the browser-source settings a streamer pastes into OBS. Preset -> exact size, fps ->
    #      the same per-frame budget as frame_budget_ms, transparent flips the CSS + URL hint, bad input refused.
    prof = obs_capture_profile(preset="1080p", fps=30)
    assert prof["width"] == 1920 and prof["height"] == 1080 and prof["fps"] == 30
    assert abs(prof["frame_budget_ms"] - round(frame_budget_ms(30), 3)) < 1e-9   # reuses the budget math
    assert prof["transparent"] is False and prof["custom_css"] == ""             # opaque: no custom CSS
    assert any("Browser" in s for s in prof["obs_steps"])                        # the human steps are present
    tp = obs_capture_profile(preset="720p", fps=60, transparent=True)
    assert tp["width"] == 1280 and tp["transparent"] and "rgba(0, 0, 0, 0)" in tp["custom_css"]
    assert tp["url"].endswith("#transparent")                                    # front-end hint for a clear colour
    for bad in (dict(preset="nope"), dict(fps=0)):                               # refuse loudly, never guess
        try:
            obs_capture_profile(**bad); assert False, "should have refused %r" % bad
        except ValueError:
            pass

    print("holographic_framebudget selftest OK (60fps->%.1fms / 30fps->%.1fms budgets; drops on a miss, climbs "
          "only after a streak (hysteresis); closed loop from ultra converges to the highest fitting level "
          "(%d) and meets the budget on %.0f%% of frames; FrameServer keeps per-session controllers; serve_frame "
          "times the render and returns the payload inline; demo_frame_payload raymarches a real SDF at the preset "
          "resolution; deterministic)"
          % (b60, b30, ctrl3.level, 100 * st["met_budget_frac"]))


if __name__ == "__main__":
    _selftest()
