"""Run each demo's head-to-head benchmark HEADLESSLY, so the numbers are known before shipping.

WHY THIS EXISTS. Every demo from here on carries a vanilla-three.js arm beside the leCore arm, in
the same browser, producing the same output. That is only a benchmark if the arms AGREE -- otherwise
it is two different programs being timed -- so the harness asserts agreement first and refuses to
report anything if they diverge.

It also prints the LOSING columns. A benchmark that shows only where it wins is an advertisement,
and the arithmetic-vs-measured distinction is kept explicit: transfer time is bytes divided by a
stated rate, never presented as something that was timed.
"""
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROBE = """
globalThis.performance = globalThis.performance || require('node:perf_hooks').performance;
globalThis.document = { getElementById: () => ({ addEventListener(){}, style:{},
  value:'harbour district', set innerHTML(v){}, get innerHTML(){ return ''; }, textContent:'' }) };
globalThis.TextEncoder = require('node:util').TextEncoder;
"""


def zeroasset(page):
    html = pathlib.Path(page).read_text(encoding="utf-8")
    three = re.search(r"<script>(var __C=.*?)</script>", html, re.S).group(1)
    body = html[html.index("const P = {"):html.index("// ---- interaction ---")]
    head, rest = body.split("// ---- three.js does the rendering")
    rest = rest[rest.index("const dummy = new THREE.Object3D();"):]
    src = (PROBE + three + "\n" + head + """
const city = new THREE.InstancedMesh(new THREE.BoxGeometry(1,1,1), new THREE.MeshBasicMaterial(), P.towers);
city.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(P.towers*3), 3);
const fieldGeo = new THREE.PlaneGeometry(220,220,220,220);
""" + rest.replace("accounting(name);", "") + """
build('harbour district');
console.log(JSON.stringify(benchmark('harbour district')));
""")
    pathlib.Path("/tmp/_bench.js").write_text(src, encoding="utf-8")
    r = subprocess.run(["node", "/tmp/_bench.js"], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("%s: probe failed\n%s" % (page, r.stderr[-400:]))
    return json.loads(r.stdout.strip().splitlines()[-1])


BENCHES = {"pages/zeroasset_three.html": zeroasset}


def main():
    ok = True
    for page, fn in BENCHES.items():
        b = fn(os.path.join(ROOT, page))
        agree = b["worst"] < 1e-5
        ok &= agree
        print("%s" % page.split("/")[-1])
        print("  arms agree                    %s (max |diff| %.2e)"
              % ("YES" if agree else "NO -- the rest is meaningless", b["worst"]))
        print("  instances                     %s" % f"{b['n']:,}")
        print("  vanilla  bytes downloaded     %.2f MB" % (b["bytesA"] / 1e6))
        print("  leCore   bytes downloaded     %d bytes" % b["bytesB"])
        print("  bytes ratio                   %sx" % f"{round(b['bytesA']/max(b['bytesB'],1)):,}")
        print("  vanilla  decode -> matrices   %.1f ms" % b["tA"])
        faster = b["tB"] <= b["tA"]
        print("  leCore   regenerate from name %.1f ms  (%.1fx %s on CPU)"
              % (b["tB"], (b["tA"] / b["tB"]) if faster else (b["tB"] / b["tA"]),
                 "faster" if faster else "SLOWER"))
        for mbps in (25, 5):
            print("  transfer at %-2d Mbps           %.0f ms vs %.2f ms   [arithmetic, not measured]"
                  % (mbps, b["bytesA"] * 8 / 1e6 / mbps * 1000, b["bytesB"] * 8 / 1e6 / mbps * 1000))
    print("\n%s" % ("ALL BENCHMARKS VALID" if ok else "ARMS DISAGREE -- FIX BEFORE QUOTING"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
