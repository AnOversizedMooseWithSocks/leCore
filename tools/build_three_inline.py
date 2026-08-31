"""Rebuild pages/vendor/three.inline.js from npm -- three.js as ONE self-contained plain script.

WHY THIS EXISTS. The demo pages must open from file:// with no network, and three.js ships only ES
modules split across three.module.min.js + three.core.min.js. An inline <script type="module"> can
run from file://, but the two MINIFIED halves cannot simply be concatenated -- both declare `e` at
top level. So each half is wrapped in its own function scope, the core's export list becomes a
return value, and the module's imports become a destructure from it.

THREE FORMS HAD TO BE HANDLED, and assuming one of each is what broke the first three attempts:
    import{...}from"./three.core.min.js"        the module pulling symbols in
    export{...}                                 a normal export list (the core has 1, the module 1)
    export{...}from"./three.core.min.js"        a RE-EXPORT, which looks like an export to a naive
                                                regex and leaves `from"..."` dangling in the output

Run: python3 tools/build_three_inline.py     (needs network access to registry.npmjs.org)
"""
import pathlib
import re
import subprocess
import sys
import tempfile

CORE = r'"\./three\.core\.min\.js"'


def _names(lst, from_core=False):
    out = []
    for item in lst.split(","):
        item = item.strip()
        if not item:
            continue
        mm = re.match(r'^([A-Za-z0-9_$]+)\s+as\s+([A-Za-z0-9_$"]+)$', item)
        a, b = (mm.group(1), mm.group(2).strip('"')) if mm else (item, item)
        out.append('"%s":%s' % (b, ('__C["%s"]' % a) if from_core else a))
    return out


def _cut(body, pattern, handler):
    """Remove every match of `pattern` and collect what the handler makes of each."""
    pairs, spans = [], []
    for m in re.finditer(pattern, body):
        spans.append((m.start(), m.end()))
        pairs += handler(m.group(1))
    out, last = [], 0
    for a, b in spans:
        out.append(body[last:a])
        last = b
    out.append(body[last:])
    return "".join(out), pairs


def build(dest="pages/vendor/three.inline.js"):
    tmp = tempfile.mkdtemp()
    subprocess.run(["npm", "pack", "three@latest"], cwd=tmp, check=True,
                   capture_output=True, text=True)
    tgz = next(pathlib.Path(tmp).glob("three-*.tgz"))
    subprocess.run(["tar", "xzf", tgz.name, "package/build/three.module.min.js",
                    "package/build/three.core.min.js", "package/package.json"],
                   cwd=tmp, check=True)
    b = pathlib.Path(tmp, "package", "build")
    core = (b / "three.core.min.js").read_text(encoding="utf-8")
    mod = (b / "three.module.min.js").read_text(encoding="utf-8")

    cbody, cpairs = _cut(core, r'export\{([^}]*)\};?', lambda g: _names(g))
    # re-exports FIRST: they also match the plain-export pattern and would be mangled by it
    mbody, repairs = _cut(mod, r'export\{([^}]*)\}from' + CORE + r';?',
                          lambda g: _names(g, from_core=True))
    mbody, mbinds = _cut(
        mbody, r'import\{([^}]*)\}from' + CORE + r';?',
        lambda g: ['%s=__C["%s"]' % ((m.group(2), m.group(1)) if m else (i.strip(), i.strip()))
                   for i in g.split(",") if i.strip()
                   for m in [re.match(r'^([A-Za-z0-9_$]+)\s+as\s+([A-Za-z0-9_$]+)$', i.strip())]])
    mbody, mpairs = _cut(mbody, r'export\{([^}]*)\};?', lambda g: _names(g))

    out = ("var __C=(function(){" + cbody + "\nreturn {" + ",".join(cpairs) + "};})();\n"
           "var THREE=(function(){var " + ",".join(sorted(set(mbinds))) + ";\n"
           + mbody + "\nreturn {" + ",".join(mpairs + repairs) + "};})();\n")
    p = pathlib.Path(dest)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(out, encoding="utf-8")

    # A build that parses but is missing a symbol fails at run time, in a browser, silently enough
    # to waste an afternoon -- so check for what the pages actually use.
    chk = subprocess.run(["node", "--check", str(p)], capture_output=True, text=True)
    if chk.returncode:
        raise SystemExit("wrapped three.js does not parse: " + chk.stderr[:300])
    need = ("WebGLRenderer Points MeshStandardMaterial WebGLRenderTarget AdditiveBlending DataTexture "
            "Scene PerspectiveCamera BufferGeometry BufferAttribute ShaderMaterial DirectionalLight "
            "Mesh PlaneGeometry Vector3 Color FloatType NearestFilter RGBAFormat ACESFilmicToneMapping "
            "SRGBColorSpace OrthographicCamera Float32BufferAttribute AmbientLight DoubleSide Vector2 "
            "PCFSoftShadowMap NoBlending").split()
    r = subprocess.run(["node", "-e",
                        "eval(require('fs').readFileSync(%r,'utf8'));"
                        "const n=%r;const m=n.filter(k=>!(k in THREE));"
                        "console.log(THREE.REVISION+'|'+(m.join(',')||'none'));" % (str(p), need)],
                       capture_output=True, text=True)
    rev, missing = (r.stdout.strip().split("|") + ["?"])[:2]
    if missing != "none":
        raise SystemExit("wrapped three.js is missing symbols the pages use: " + missing)
    print("three r%s -> %s (%.0f KB), all %d needed symbols present"
          % (rev, dest, p.stat().st_size / 1024, len(need)))
    return dest


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "pages/vendor/three.inline.js")
