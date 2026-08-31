# A reusable page checker. Written to a FILE because the previous run put it in a shell -c string
# and the backslash-escaped ${N} never matched -- so it reported the cloth shaders BROKEN when they
# compile fine. Harness, not artifact, for the fourth time today; shell quoting is a delimiter trap
# exactly like the backtick and the apostrophe were.
import ctypes, re, subprocess, sys
sys.path.insert(0, "research/shader_retrieval")
import glsl_es_compile as E

# cloth_three.html mixes three.js ShaderMaterial bodies (which three prefixes at compile time and a
# bare compiler cannot validate) with ONE raw GLSL 300 es program -- leCore's scatter kernel. Only
# the raw pair is checkable here, and it is the one that matters.
# cloth_three.html's solver is a three.js RawShaderMaterial with glslVersion: GLSL3. THREE PREPENDS
# ITS OWN DEFINES to every material, so the source must NOT carry a `#version` line -- three emits
# it. To be worth anything this check must compile the shader in THAT environment, prologue and all;
# compiling it standalone with #version first is what let a broken page ship.
RAW_PAIRS = {"pages/cloth_three.html": ("LECORE_KERNEL_VS", "LECORE_KERNEL_FS"),
             "pages/zeroasset_three.html": ("LECORE_GEN_VS", "LECORE_GEN_FS"),
             "pages/volume_three.html": ("LECORE_VOL_VS", "LECORE_VOL_FS"),
             "pages/volume_three.html#sky": ("LECORE_SKY_VS", "LECORE_SKY_FS"),
             "pages/volume_three.html#comp": ("LECORE_COMP_VS", "LECORE_COMP_FS"),
             "pages/volume_three.html#occ": ("LECORE_OCC_VS", "LECORE_OCC_FS")}

# Pages with no raw GLSL of their own: three.js supplies every shader. What must hold is that the
# script parses and that NOTHING is fetched -- for zeroasset_three the whole claim is that no scene
# data is downloaded, so a stray network reference would falsify the headline, not just slow it.
JS_ONLY = []

# Pages that patch THREE'S OWN vertex shader via onBeforeCompile. Those replacements are never seen
# by a RawShaderMaterial check, and three splices them into its chunk order -- <beginnormal_vertex>
# BEFORE <begin_vertex> -- so a variable declared in the wrong one is an undeclared identifier at
# runtime and nowhere else. Reconstruct that order and compile it.
PATCH_PAGES = {"pages/zeroasset_three.html": ["beginnormal_vertex", "begin_vertex"]}
THREE_PROLOGUE = ("#version 300 es\n"
                  "#define SHADER_TYPE RawShaderMaterial\n"
                  "#define SHADER_NAME \n")

PAGES = {
    "pages/field_demo.html": ((("VS_QUAD","FS_DRIFT"),("VS_SPLAT","FS_SPLAT"),
                               ("VS_QUAD","FS_DIFFUSE"),("VS_QUAD","FS_BLIT")), None),
    "pages/cloth_demo.html": ((("VSQ","FS_PREDICT"),("VS_SCATTER","FS_SCATTER"),
                               ("VSQ","FS_APPLY"),("VS_DRAW","FS_DRAW"),
                               ("VS_ARROW","FS_ARROW")), "auto"),
}

def interp(s, n):
    """Substitute exactly what the browser substitutes. A checker that validates un-interpolated
    template text is validating source no GL compiler will ever see -- it reported four syntax
    errors that did not exist and hid a real precision bug behind them."""
    if n is None:
        return s
    s = (s.replace("${N - 1}", str(n-1)).replace("${N-1}", str(n-1)).replace("${N}", str(n)))
    # any remaining ${expr} is a numeric literal the page computes; evaluate the safe ones
    def num(m):
        e = m.group(1)
        try:
            return "%.8f" % eval(e, {"__builtins__": {}}, {"N": n, "SP": 0.9/(n-1)})
        except Exception:
            return m.group(0)
    return re.sub(r"\$\{([^}]*)\}", num, s)

E.make_context()
allok = True
for page, (pairs, n) in PAGES.items():
    html = open(page, encoding="utf-8").read()
    if n == "auto":                       # read the grid size from the page, never assume it
        m = re.search(r"const N\s*=\s*(\d+)", html)
        n = int(m.group(1)) if m else None
    src = {m.group(1): interp(m.group(2), n)
           for m in re.finditer(r"const (\w+)\s*=\s*`(#version[^`]*)`", html, re.S)}
    missing = [x for pair in pairs for x in pair if x not in src]
    if missing:
        # A checker that cannot find the shaders it was told to check must SAY SO, not KeyError and
        # not quietly pass. This is the harness failing, and it reports as a failure.
        print("%-26s CANNOT FIND: %s (found %s)" % (page.split("/")[-1], missing, sorted(src)))
        allok = False
        continue
    left = [k for k, v in src.items() if "${" in v]
    ok = not left
    for v, f in pairs:
        a, la, vid = E.compile_shader(E.GL_VERTEX_SHADER, src[v])
        b, lb, fid = E.compile_shader(E.GL_FRAGMENT_SHADER, src[f])
        st = ctypes.c_int()
        if a and b:
            pid = E.gl.glCreateProgram(); E.gl.glAttachShader(pid, vid); E.gl.glAttachShader(pid, fid)
            E.gl.glLinkProgram(pid); E.gl.glGetProgramiv(pid, E.GL_LINK_STATUS, ctypes.byref(st))
        if not (a and b and st.value):
            ok = False
            print("  %s  %s/%s: %s" % (page, v, f, (la or lb or "link failed")[:180]))
    open("/tmp/_p.js", "w").write(re.search(r"<script>([\s\S]*)</script>", html).group(1))
    js = subprocess.run(["node", "--check", "/tmp/_p.js"], capture_output=True, text=True).stderr.strip()
    print("%-26s shaders %-7s js %s%s" % (page.split("/")[-1], "LINK" if ok else "BROKEN",
          "ok" if not js else js[:60], "" if not left else "  UNSUBSTITUTED: %s" % left))
    allok &= ok and not js
for page, (vk, fk) in RAW_PAIRS.items():
    html = open(page.split("#")[0], encoding="utf-8").read()
    m = re.search(r'"N":(\d+)', html)
    n = int(m.group(1)) if m else None      # not every page templates a grid size
    sub = (lambda t: t) if n is None else \
          (lambda t: t.replace("${N-1}", str(n-1)).replace("${N}", str(n)))
    def literal(marker):
        # the whole template literal CONTAINING the marker, wherever the marker sits inside it
        m = re.search(r'`([^`]*' + marker + r'[^`]*)`', html, re.S)
        if not m:
            raise SystemExit("%s: cannot find the shader marked %s" % (page, marker))
        return m.group(1)
    vraw, fraw = literal(vk), literal(fk)
    vs, fs = THREE_PROLOGUE + sub(vraw), THREE_PROLOGUE + sub(fraw)
    # NOTE ON BACKTICKS IN SHADER BODIES (instrument error 33, paid twice): a stray backtick --
    # even inside a GLSL comment -- closes the JS template literal and truncates the shader. There
    # is deliberately NO dedicated check for it here, because the extractor's own regex
    # `([^`]*MARKER[^`]*)` cannot return a body containing one. The COMPILE STAGE BELOW catches it
    # instead, and was verified doing so: a page with a backtick reinstated reports
    # "leCore kernel BROKEN ... syntax error, unexpected end of file" plus a JS parse failure.
    if "#version" in vraw or "#version" in fraw:
        print("%-26s SHADER CARRIES ITS OWN #version -- three.js prepends defines, so it will be "
              "rejected" % page.split("/")[-1])
        allok = False
        continue
    a, la, vid = E.compile_shader(E.GL_VERTEX_SHADER, vs)
    b, lb, fid = E.compile_shader(E.GL_FRAGMENT_SHADER, fs)
    st = ctypes.c_int()
    if a and b:
        pid = E.gl.glCreateProgram(); E.gl.glAttachShader(pid, vid); E.gl.glAttachShader(pid, fid)
        E.gl.glLinkProgram(pid); E.gl.glGetProgramiv(pid, E.GL_LINK_STATUS, ctypes.byref(st))
    ok = bool(a and b and st.value)
    scripts = re.findall(r"<script>([\s\S]*?)</script>", html)
    open("/tmp/_p.js", "w").write(scripts[-1])
    js = subprocess.run(["node", "--check", "/tmp/_p.js"], capture_output=True, text=True).stderr.strip()
    net = len(re.findall(r'src="https?://', html))
    print("%-26s leCore kernel %-7s js %-14s network refs %d"
          % (page.split("/")[-1], "LINK" if ok else "BROKEN", "ok" if not js else js[:12], net))
    if not ok:
        print("   ", (la or lb or "link failed")[:200])
    allok &= ok and not js and net == 0

for page, order in PATCH_PAGES.items():
    html = open(page, encoding="utf-8").read()
    fn = html[html.index("function patchVertex"):html.index("const mat = new THREE")]
    reps, missing = {}, []
    for key in order:
        m = re.search(r'\.replace\("#include <' + key + r'>", `([\s\S]*?)`\)', fn)
        if m:
            reps[key] = re.sub(r"\$\{withColor \? \"(.*?)\" : \"(.*?)\"\}", r"\1", m.group(1))
        else:
            missing.append(key)
    # EACH BLOCK MUST COMPILE ON ITS OWN. three does not guarantee both are included: depth_vert
    # wraps <beginnormal_vertex> in #ifdef USE_DISPLACEMENTMAP, so a block that depends on its
    # neighbour having run is broken in the shadow pass and nowhere else. Compiling them TOGETHER
    # is the assumption that let exactly that reach a browser.
    PRE = ("#version 300 es\nprecision highp float; precision highp int;\n"
           "uniform sampler2D uXform, uScl, uCol; uniform int uGW;\n"
           "out vec3 vTowerColor;\nin vec3 position; in vec3 normal;\n"
           "uniform mat4 modelViewMatrix, projectionMatrix;\n"
           "ivec2 lcCell(){ return ivec2(gl_InstanceID % uGW, gl_InstanceID / uGW); }\n")
    okc, log = True, ""
    for key in order:
        tail = ("\n  gl_Position = projectionMatrix*modelViewMatrix*vec4(transformed,1.0);\n"
                if key == "begin_vertex" else
                "\n  gl_Position = vec4(objectNormal, 1.0);\n")
        ok1, lg, _ = E.compile_shader(E.GL_VERTEX_SHADER,
                                      PRE + "void main(){\n" + reps.get(key, "") + tail + "}\n")
        if not ok1:
            okc = False
            log = "[%s] %s" % (key, lg)
            break
    print("%-26s onBeforeCompile patch %s%s"
          % (page.split("/")[-1], "COMPILES" if okc else "FAILS",
             "" if not missing else "  MISSING CHUNKS: %s" % missing))
    if not okc:
        print("   ", log[:220])
    allok &= okc and not missing

for page in JS_ONLY:
    html = open(page, encoding="utf-8").read()
    scripts = re.findall(r"<script>([\s\S]*?)</script>", html)
    open("/tmp/_p.js", "w").write(scripts[-1])
    js = subprocess.run(["node", "--check", "/tmp/_p.js"], capture_output=True, text=True).stderr.strip()
    net = len(re.findall(r'src="https?://', html))
    print("%-26s js %-14s network refs %d" % (page.split("/")[-1], "ok" if not js else js[:12], net))
    if js:
        print("   ", js[:200])
    allok &= (not js) and net == 0

# ${...} INSIDE DOUBLE QUOTES IS LITERAL TEXT IN JS. It only interpolates inside backticks, so a
# hole in a quoted string reaches the reader verbatim -- caught exactly that on volume_three before
# it shipped.
for page in sorted({k.split("#")[0] for k in RAW_PAIRS} | set(PAGES)):
    html = open(page, encoding="utf-8").read()
    own = html[html.rindex("<script>"):]
    # Remove template literals first: a ${...} inside backticks is correct, and only a hole in a
    # plain quoted string is a defect. Matching on the quotes alone flagged row()'s own template.
    # A // COMMENT THAT STILL HAS DECLARATIONS AFTER IT ON THE SAME LINE silently deletes them.
    # Cost three uniforms and two symptoms (a TypeError and a cloud with no self-shadowing) before
    # anything in the toolchain noticed, because the page still parses perfectly.
    for ln in own.splitlines():
        c = ln.find("//")
        if c >= 0 and ":{value" in ln[c:]:
            print("%-26s COMMENT SWALLOWS A UNIFORM DECLARATION: %s"
                  % (page.split("/")[-1], ln.strip()[:70]))
            allok = False
    outside = re.sub(r"`[^`]*`", "``", own, flags=re.S)
    holes = [m.group(0) for m in re.finditer(r'"[^"\n]*\$\{[^}]*\}[^"\n]*"', outside)]
    if holes:
        print("%-26s TEMPLATE HOLE IN A QUOTED STRING: %s" % (page.split("/")[-1], holes[0][:70]))
        allok = False

print("\n%s" % ("ALL PAGES OK" if allok else "PROBLEMS ABOVE"))
