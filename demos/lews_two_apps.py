"""Two apps, one workspace: a PAINTER and a MODELLER as separate processes on the same .lews directory.

The painter writes a `lecore.image` texture and repaints it three times. The modeller owns a `lecore.mesh` + a
`lecore.scene` binding that texture to the mesh; it polls `changes_since(rev)`, and each time the painter's section
changes it re-renders the textured mesh with the engine (a UV-mapped flat-shaded preview) and saves a frame. Neither
process knows the other exists beyond the workspace; a third app could join and see both.
    python3 demos/lews_two_apps.py <workspace_dir>
"""
import os, sys, time, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


def painter(root):
    import lecore
    from holographic.io_and_interop.holographic_container import image_section
    m = lecore.UnifiedMind(dim=64, seed=0)
    w = m.lews_open(root, app="painter", app_version="demo")
    for i, colour in enumerate([(0.9, 0.2, 0.2), (0.2, 0.8, 0.3), (0.2, 0.3, 0.9)]):
        img = np.zeros((64, 64, 4), np.float32); img[..., 3] = 1
        yy, xx = np.mgrid[0:64, 0:64]; chk = ((xx // 8 + yy // 8) % 2).astype(np.float32)
        img[..., :3] = np.asarray(colour) * (0.55 + 0.45 * chk[..., None])
        s = image_section(img, name="paint_%d" % i); s["id"] = "tex"
        rev = w.put(s)
        print("painter: put tex rev %d (%s)" % (rev, colour), flush=True)
        time.sleep(0.6)


def modeller(root, frames):
    import lecore
    m = lecore.UnifiedMind(dim=64, seed=0)
    w = m.lews_open(root, app="modeller", app_version="demo")
    V = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], np.float32); F = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    w.put(m.lews_mesh_section(V, F, sid="quad", name="quad", uv=UV))
    w.put(m.lews_scene_section([{"id": "o1", "mesh": "quad", "texture": "tex", "material": None}]))
    seen = w.rev(); got = 0; deadline = time.time() + 20
    while got < frames and time.time() < deadline:
        ch = w.wait_for_change(seen, timeout=2.0)
        for e in ch:
            seen = max(seen, e["rev"])
            if e["id"] == "tex" and e["op"] == "put":
                tex = w.get("tex")["arrays"]["image"]
                # the engine's own UV sampling: a 96x96 view of the quad with the painter's texture, nearest-neighbour
                uu, vv = np.meshgrid(np.linspace(0, 1, 96), np.linspace(0, 1, 96))
                px = tex[np.clip((vv * (tex.shape[0] - 1)).astype(int), 0, tex.shape[0] - 1),
                         np.clip((uu * (tex.shape[1] - 1)).astype(int), 0, tex.shape[1] - 1), :3]
                m.save_render(os.path.join(root, "modeller_frame_%d.png" % got), np.clip(px, 0, 1))
                got += 1
                print("modeller: texture rev %d from %s -> frame %d" % (e["rev"], e["app"], got - 1), flush=True)
    print("modeller: done, %d frames; journal has %d entries" % (got, len(w.changes_since(0))), flush=True)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lews_demo"
    role = sys.argv[2] if len(sys.argv) > 2 else None
    if role == "painter":
        painter(root)
    elif role == "modeller":
        modeller(root, frames=3)
    else:
        os.makedirs(root, exist_ok=True)
        env = dict(os.environ, PYTHONHASHSEED="0")
        pm = subprocess.Popen([sys.executable, __file__, root, "modeller"], env=env)
        time.sleep(3.0)                                            # let the modeller publish its mesh + scene first
        pp = subprocess.Popen([sys.executable, __file__, root, "painter"], env=env)
        pp.wait(); pm.wait()
        import lecore
        print(json.dumps(lecore.UnifiedMind(dim=64, seed=0).lews_describe(root), indent=1)[:600])
