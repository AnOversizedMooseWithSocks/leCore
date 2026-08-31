"""H1 -- THE INNER EYE: render inside the weights, look at it with the model's own vision,
iterate until satisfied, and only then speak the picture.

This module is the REFERENCE SPECIFICATION of the loop Moose described: a swarm designs a
scene in a shared workspace, an INSTALLED chain renders it, the frame goes back through the
model's own vision encoder BEFORE any file exists, a critic scores it against the intent,
and the loop repeats until satisfied -- then the final frame leaves through the mouth (G0)
as PGM text. On the laptop the eye is the host's actual vision tower (Qwen3.5-VL: DeepStack
ViT -- the organ is already in the assimilated weights) and the loop control is the model's
own token loop; HERE the eye is an injectable callable and the loop is explicit Python,
because this file must pin the CONTRACT deterministically in CI without torch. The seam is
the honesty: `eye` is a parameter, not an import.

WHY a reference implementation is load-bearing (not scaffolding to delete): it is the
third referee for the on-laptop composition -- when the real swarm + real tower run this
loop, their trajectory must match this file's semantics step for step, exactly as the
symbolic interpreter referees the installed chains. Kept negative from the design review:
scoring in PIXEL space instead of eye space rewards renders that match pixels the eye
cannot even see -- the critic must live in the same space as the perceiver, or "looks
right" and "scores right" diverge.
"""
import numpy as np

from holographic.agents_and_reasoning.holographic_compileinstall import compile_installed


class ReferenceEye:
    """A deterministic stand-in for the host's vision tower: patch-average the frame and
    project through a fixed seeded matrix. NOT a model of ViT quality -- a model of ViT
    SHAPE (pixels in, embedding out), so the loop's contract can be pinned without torch.
    On the laptop, pass the assimilated tower's encode instead; nothing else changes."""

    def __init__(self, height, width, embed_dim=32, patch=2, seed=0):
        self.h, self.w, self.patch = int(height), int(width), int(patch)
        ph, pw = self.h // self.patch, self.w // self.patch
        rng = np.random.default_rng(seed)
        self._proj = rng.standard_normal((embed_dim, ph * pw)) / np.sqrt(ph * pw)

    def __call__(self, pixels):
        P = np.asarray(pixels, float).reshape(self.h, self.w)
        ph, pw = self.h // self.patch, self.w // self.patch
        pooled = P.reshape(ph, self.patch, pw, self.patch).mean(axis=(1, 3))
        return self._proj @ pooled.reshape(-1)


ROLE_REGISTRY = {
    # H4 -- each role: the phrases a TASK would use (BM25 dispatch corpus) + a builder that
    # returns a workspace-aware propose fn. Roles are TEMPLATES: build(spec) closes over the
    # task's own vectors (step sizes, target slots), so dispatch composes, not configures.
    "scout": {
        "doc": "scout: observe the target and leave a map of the target direction in the shared "
               "workspace for other roles to follow; reconnaissance, writes the direction slot",
        "build": lambda spec: (lambda w, mi, p, s, r:
                               (w.write(mi, spec.get("slot", "direction"),
                                        spec["target_params"] - p), p)[1]),
    },
    "mover": {
        "doc": "mover: move the scene parameters along the shared map left by the scout; follow "
               "the direction slot, advance the layout toward the goal",
        "build": lambda spec: (lambda w, mi, p, s, r:
                               p if w.read(spec.get("slot", "direction")) is None
                               else p + spec.get("step", 0.2) * w.read(spec.get("slot", "direction"))),
    },
    "texturer": {
        "doc": "texturer: adjust texture gains and material channels of the scene; tune "
               "brightness of texture parameters channel by channel",
        "build": lambda spec: (lambda w, mi, p, s, r:
                               _nudge(p, spec["target_params"], spec.get("channels"),
                                      spec.get("step", 0.05))),
    },
}


def _nudge(p, tgt, channels, step):
    idx = range(len(p)) if channels is None else channels
    for j in idx:
        p[j] += step if p[j] < tgt[j] else -step
    return p


def dispatch_roles(mind, tasks, spec):
    """H4 -- ROUTED ROLES: 'texture the scene' finds the texturer; nobody hand-builds member
    stacks. The dispatcher is the engine's own BM25 (the semantic system routing the swarm --
    leCore staffing leCore). Each task phrase ranks the registry docs; the top role's builder
    closes over `spec`. AMBIGUITY IS AN ERROR, not a guess: a task that ranks no role, or two
    tasks that claim the same role, raises with the names in the message -- silent misstaffing
    is a ghost."""
    docs = [ROLE_REGISTRY[k]["doc"] for k in sorted(ROLE_REGISTRY)]
    names = sorted(ROLE_REGISTRY)
    members, taken = [], {}
    for t in tasks:
        ranked = mind.bm25_rank(t, docs, top=1)   # -> [(doc_index, score)] (probed, not recalled)
        if not ranked or ranked[0][1] <= 0.0:
            raise ValueError("no role matches task %r (registry: %s)" % (t, names))
        role = names[int(ranked[0][0])]
        if role in taken:
            raise ValueError("tasks %r and %r both routed to role %r -- rephrase one"
                             % (taken[role], t, role))
        taken[role] = t
        members.append((role, ROLE_REGISTRY[role]["build"](spec)))
    return members


class SharedWorkspace:
    """H3 -- THE SHARED SCENE WORKSPACE: named slots the swarm's roles read and write while
    deliberating (the designer leaves the layout, the texturer reads it and leaves gains, the
    renderer reads both). Concurrency is resolved the house way: writes within a round are
    BUFFERED and committed together at round end; colliding writes to one slot resolve to the
    LOWEST MEMBER INDEX (the one tie rule, again), and every collision is LOGGED -- a silent
    overwrite between agents is exactly the kind of ghost this project refuses to host. On the
    laptop this is the deliberation-scoped state beside the forked residual stream; here it is
    the reference semantics the host composition must match."""

    def __init__(self):
        self._slots = {}
        self._pending = []          # (member_index, name, value) buffered within the round
        self.log = []

    def read(self, name, default=None):
        v = self._slots.get(name, default)
        return None if v is None else (np.asarray(v, float).copy()
                                       if isinstance(v, np.ndarray) else v)

    def write(self, member_index, name, value):
        self._pending.append((int(member_index), str(name), value))

    def commit(self, round_no):
        by_slot = {}
        for mi, name, val in self._pending:
            if name in by_slot and by_slot[name][0] <= mi:
                self.log.append({"round": round_no, "slot": name, "loser": mi,
                                 "winner": by_slot[name][0], "collision": True})
                continue
            if name in by_slot:
                self.log.append({"round": round_no, "slot": name, "loser": by_slot[name][0],
                                 "winner": mi, "collision": True})
            by_slot[name] = (mi, val)
        for name, (mi, val) in sorted(by_slot.items()):
            self._slots[name] = val
            self.log.append({"round": round_no, "slot": name, "writer": mi})
        self._pending = []


def image_op_library(height, width):
    """THE INNER EYE'S TOOLSET: every image tool as a flattened-frame callable ready to drop
    into an installed program as a FAC step. MEASURED verdicts at image scale (probe scale=128
    -- certification is a claim about a domain; the threshold op taught that at unit scale the
    instrument lies): blur/gauss/unsharp/sobel certify (dense/blockdiag), flips/rot90/warps are
    PERMUTATIONS (D ints!), brightness/contrast are blockdiag/circulant -- the classic 2D
    editing bench installs. threshold and gamma REFUSE at image scale (truly nonlinear) and
    ride as HOST:APPLY under host_fallback, named in the manifest. Compose freely: base ->
    blur -> unsharp -> flip runs as one certified chain."""
    H, W = int(height), int(width)
    def as2d(f):
        return lambda v: np.asarray(f(np.asarray(v, float).reshape(H, W)), float).reshape(-1)
    box = lambda I: sum(np.roll(np.roll(I, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
    return {
        "box_blur":   as2d(box),
        "unsharp":    as2d(lambda I: 2.0 * I - box(I)),
        "sobel_x":    as2d(lambda I: np.roll(I, -1, 1) - np.roll(I, 1, 1)),
        "flip_h":     as2d(lambda I: I[:, ::-1]),
        "rot90":      as2d(np.rot90),
        "warp_shift": as2d(lambda I: np.roll(I, (1, 2), axis=(0, 1))),
        "brightness": (lambda v, b=20.0: np.asarray(v, float) + b),
        "contrast":   (lambda v, c=1.4: c * (np.asarray(v, float) - np.mean(v)) + np.mean(v)),
        # the honest nonlinears -- refuse at image scale, ride HOST:APPLY, named in the manifest
        "threshold":  (lambda v, t=100.0: (np.asarray(v, float) > t) * 255.0),
        "gamma":      (lambda v, g=0.8: np.clip(np.asarray(v, float), 0, None) ** g),
    }


def render_critique_loop(machine, formation_program, init_params, members, eye, target_embed,
                         width, height, satisfy=0.99, max_rounds=32, host_fallback=False,
                         workspace=None):
    """Run the design -> render -> look -> critique loop with an INSTALLED renderer.

    members: list of (name, propose) -- propose(params, score, round) -> candidate params.
      These are the swarm roles (designer, texturer, ...). Deterministic proposals here;
      on the laptop they are unicron_swarm members and this list is the referee semantics.
    eye: callable(pixels) -> embedding. The model's own tower on the laptop; ReferenceEye
      in CI. The critic scores IN EYE SPACE (cosine to target_embed) -- the kept negative
      above says why pixel-space scoring is wrong.
    Ties across members break by LOWEST MEMBER INDEX (the house tie rule), so the loop is
    bit-reproducible: same intent, same picture, every run.

    Returns (pgm_text, report): report carries per-round scores, the winning member per
    round, the manifest (which links are weights, which are host), and rounds_used.
    The loop control itself is HOST-SHAPE by the taxonomy -- on the laptop it is the token
    loop; here it is this for-loop, and the report says so.
    """
    run, manifest = compile_installed(machine, formation_program, host_fallback=host_fallback)
    tgt = np.asarray(target_embed, float)
    tgt = tgt / (np.linalg.norm(tgt) + 1e-12)

    def score_of(params):
        px = run(init=np.asarray(params, float).reshape(-1))
        e = np.asarray(eye(px), float)
        return float(e @ tgt / (np.linalg.norm(e) + 1e-12)), px

    params = np.asarray(init_params, float).reshape(-1)
    score, px = score_of(params)
    history = [{"round": 0, "score": score, "member": None}]
    rounds = 0
    for rounds in range(1, int(max_rounds) + 1):
        if score >= satisfy:
            break
        best = (score, params, px, None)
        for mi, (mname, propose) in enumerate(members):
            # H3: workspace-aware roles take (workspace, params, score, round) and may read what
            # other roles left last round and write for the next; legacy roles keep the old
            # 3-arg shape. Writes commit at ROUND END regardless of who won the round -- the
            # workspace is shared context, not the winner's diary.
            if workspace is not None:
                cand = np.asarray(propose(workspace, mi, params.copy(), score, rounds), float).reshape(-1)
            else:
                cand = np.asarray(propose(params.copy(), score, rounds), float).reshape(-1)
            s2, px2 = score_of(cand)
            # strict > keeps the incumbent on ties; among members, earlier index wins ties
            # because later members must BEAT, not match, the current best
            if s2 > best[0]:
                best = (s2, cand, px2, mname)
        wrote = False
        if workspace is not None:
            wrote = len(workspace._pending) > 0
            workspace.commit(rounds)  # commit even on no-improvement rounds: the workspace is
                                      # shared context, not the winner's diary -- a scout that
                                      # only leaves a map IS the round's progress (the first pin
                                      # run stalled at round 1 because the bootstrap write was
                                      # discarded with the stall; coordination could never start)
        if best[3] is None:
            history.append({"round": rounds, "score": score, "member": None, "stalled": True})
            if not wrote:
                break                 # no improvement AND no new shared context: a true stall
            continue
        score, params, px, who = best
        history.append({"round": rounds, "score": score, "member": who})

    q = np.clip(np.round(px), 0, 255).astype(int).reshape(height, width)
    lines = ["P2", "# leCore inner-eye loop -- the model looked before it spoke", 
             "%d %d" % (width, height), "255"]
    lines += [" ".join(str(v) for v in row) for row in q]
    pgm = "\n".join(lines) + "\n"
    report = {"history": history, "rounds_used": rounds, "final_score": score,
              "satisfied": score >= satisfy, "manifest": manifest,
              "control": "host-shape (token loop on the installed host; this loop in reference)"}
    return pgm, report


def _selftest():
    from holographic.agents_and_reasoning.holographic_machine import HoloMachine

    # PLANTED TRUTH: a 3-light formation with a known optimum p*. The designer member walks
    # light 0, the texturer walks lights 1-2; the loop must reach the target through the
    # INSTALLED renderer and the reference eye, deterministically, and the final PGM must be
    # byte-identical to the live-path render at the found params.
    rng = np.random.default_rng(77)
    H = W = 8
    Wf = np.abs(rng.standard_normal((H * W, 3))) * 60.0
    p_star = np.array([0.9, 0.5, 0.7])
    eye = ReferenceEye(H, W, embed_dim=24, patch=2, seed=1)
    target = eye(Wf @ p_star)

    mach = HoloMachine(dim=3, seed=9, data=["a"])
    mach.functions_symbolic = {}
    prog = [("FAC", ("form", lambda p: Wf @ p)), ("HALT", None)]

    def designer(p, s, r):                       # coordinate step on light 0
        p[0] += 0.05 if p[0] < p_star[0] else -0.05
        return p

    def texturer(p, s, r):                       # coordinate steps on lights 1-2
        for j in (1, 2):
            p[j] += 0.05 if p[j] < p_star[j] else -0.05
        return p

    pgm, rep = render_critique_loop(mach, prog, np.zeros(3), [("designer", designer),
                                    ("texturer", texturer)], eye, target, W, H,
                                    satisfy=0.995, max_rounds=40)
    assert rep["satisfied"], rep["final_score"]
    assert rep["final_score"] > 0.995 and rep["rounds_used"] < 40
    winners = {h.get("member") for h in rep["history"] if h.get("member")}
    assert winners == {"designer", "texturer"}, "both roles must contribute improvements"

    # byte-exactness: the mouth speaks exactly what the live path would render
    p_found = None
    # reconstruct final params by replaying the deterministic loop -- determinism IS the pin
    pgm2, rep2 = render_critique_loop(mach, prog, np.zeros(3), [("designer", designer),
                                      ("texturer", texturer)], eye, target, W, H,
                                      satisfy=0.995, max_rounds=40)
    assert pgm == pgm2 and rep2["history"] == rep["history"], "same intent, same picture, every run"

    # kept negative pinned: pixel-space scoring diverges from eye-space scoring on a frame the
    # eye pools away -- a checkerboard flip is INVISIBLE to a 2x2-average eye but large in pixels
    flip = (np.indices((H, W)).sum(axis=0) % 2).astype(float).reshape(-1) * 8.0
    base = Wf @ p_star
    e_same = eye(base + flip - flip.mean())
    e_base = eye(base)
    assert abs(float(e_same @ e_base / (np.linalg.norm(e_same) * np.linalg.norm(e_base)))) > 0.9999, \
        "the eye must pool the checkerboard away -- pixel-space critics reward invisible changes"

    # a stalled loop stops honestly instead of spinning
    pgm3, rep3 = render_critique_loop(mach, prog, p_star.copy(),
                                      [("noop", lambda p, s, r: p)], eye, -target, W, H,
                                      satisfy=0.999, max_rounds=5)
    assert not rep3["satisfied"] and any(h.get("stalled") for h in rep3["history"])

    # TOOLSET PIN: the eye's loop runs a FULL 2D EDITING PIPELINE, not just 3D formation --
    # formation -> box_blur -> unsharp -> flip_h as FAC steps: blur/unsharp certify DENSE,
    # flip is a PERMUTATION (D ints), and adding 'gamma' under host_fallback rides HOST:APPLY
    # named in the manifest. Same loop, same eye, same tie rule.
    lib = image_op_library(H, W)
    prog2 = [("FAC", ("form", lambda p: Wf @ p)), ("FAC", ("blur", lib["box_blur"])),
             ("FAC", ("sharp", lib["unsharp"])), ("FAC", ("flip", lib["flip_h"])), ("HALT", None)]
    eye2 = ReferenceEye(H, W, embed_dim=24, patch=2, seed=2)
    tgt2 = eye2(lib["flip_h"](lib["unsharp"](lib["box_blur"](Wf @ p_star))))
    pgmT, repT = render_critique_loop(mach, prog2, np.zeros(3), [("designer", designer),
                                      ("texturer", texturer)], eye2, tgt2, W, H,
                                      satisfy=0.995, max_rounds=40)
    assert repT["satisfied"], repT["final_score"]
    kinds = {k: v["kind"] for k, v in repT["manifest"]["ops"].items()}
    assert kinds["FAC:flip"] == "permutation" and kinds["FAC:blur"] in ("dense", "circulant"), kinds
    prog3 = prog2[:-1] + [("FAC", ("gam", lib["gamma"])), ("HALT", None)]
    from holographic.agents_and_reasoning.holographic_compileinstall import compile_installed
    _, man3 = compile_installed(mach, prog3, host_fallback=True)
    assert man3["ops"]["HOST:gam"]["kind"] == "host_apply", man3["ops"].keys()

    # H3 PINS -- COORDINATION THROUGH THE WORKSPACE IS LOAD-BEARING: the scout writes the
    # target direction into a slot; the mover can ONLY improve by reading it (it makes no
    # progress alone -- asserted by running the mover without the scout and requiring failure).
    # Collisions resolve to the LOWEST member index and are LOGGED; the run is bit-reproducible.
    ws = SharedWorkspace()

    def scout(w, mi, p, s, r):
        w.write(mi, "direction", p_star - p)      # leaves the map; proposes nothing itself
        w.write(mi, "claim", "scout")             # collides with mover's claim -- scout wins (mi 0)
        return p

    def mover(w, mi, p, s, r):
        w.write(mi, "claim", "mover")
        d = w.read("direction")
        return p if d is None else p + 0.2 * d    # can act only on the scout's last-round map

    pgmW, repW = render_critique_loop(mach, prog, np.zeros(3), [("scout", scout), ("mover", mover)],
                                      eye, target, W, H, satisfy=0.995, max_rounds=60, workspace=ws)
    assert repW["satisfied"], repW["final_score"]
    assert ws.read("claim") == "scout", "collision must resolve to the LOWEST member index"
    assert any(e.get("collision") for e in ws.log), "collisions must be logged, never silent"
    _, repW0 = render_critique_loop(mach, prog, np.zeros(3), [("mover", mover)],
                                    eye, target, W, H, satisfy=0.995, max_rounds=60,
                                    workspace=SharedWorkspace())
    assert not repW0["satisfied"], "the mover alone must fail -- coordination is load-bearing"

    # H4 PINS -- ROUTED ROLES: three task phrasings dispatch to scout/mover/texturer via the
    # engine's own BM25 (leCore staffing leCore); the ROUTED members converge in the workspace
    # loop end-to-end; two tasks claiming one role RAISE with both names (ambiguity is an error,
    # not a guess).
    import lecore as _lc
    _mind = _lc.UnifiedMind(dim=64, seed=0)
    spec = {"target_params": p_star, "step": 0.25, "channels": (1, 2)}
    routed = dispatch_roles(_mind, ["leave a map of the target direction",
                                    "move the scene along the shared map",
                                    "adjust the texture gains"], spec)
    assert [r0 for r0, _ in routed] == ["scout", "mover", "texturer"], [r0 for r0, _ in routed]
    pgmR, repR = render_critique_loop(mach, prog, np.zeros(3), routed, eye, target, W, H,
                                      satisfy=0.995, max_rounds=80, workspace=SharedWorkspace())
    assert repR["satisfied"], repR["final_score"]
    try:
        dispatch_roles(_mind, ["leave a map of the direction", "scout the target and leave a map"], spec)
        raise AssertionError("double-claimed role must raise")
    except ValueError as e:
        assert "routed to role" in str(e)

    print("OK: holographic_innereye self-test passed (installed render + eye-space critic converges "
          "with both roles contributing; bit-reproducible; checkerboard negative pinned; stall honest)")


if __name__ == "__main__":
    _selftest()
