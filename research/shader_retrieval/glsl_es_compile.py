"""Compile the EXACT WebGL2 shaders (#version 300 es) with a real GLES 3.0 compiler.

WHY THIS AND NOT THE 330-core RUN: llvmpipe gave us desktop GLSL 330 core, and the standing
caveat was "written to the WebGL2 subset is not run as WebGL2". WebGL2's shading language is
GLSL ES 3.00, a DIFFERENT language with its own rules (precision qualifiers are mandatory for
float in fragment shaders, `out` declarations, no implicit conversions, integer texelFetch
rules). Mesa can hand out a surfaceless GLES 3.0 context over EGL, so the ES compiler itself
can be asked whether the text is legal -- which is the part a browser would also do first.

This does NOT run in a browser. It runs the browser's SHADING LANGUAGE through the same Mesa
front end a browser's ANGLE/Mesa path would use. The residual gap is stated at the end.
"""
import ctypes, ctypes.util, sys

EGL_SUCCESS = 0x3000
EGL_PLATFORM_SURFACELESS_MESA = 0x31DD
EGL_OPENGL_ES_API = 0x30A0
EGL_NO_SURFACE = ctypes.c_void_p(0)
EGL_NONE = 0x3038
EGL_CONTEXT_CLIENT_VERSION = 0x3098
EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT = 0x3040, 0x00000040
EGL_SURFACE_TYPE, EGL_PBUFFER_BIT = 0x3033, 0x0001

GL_VERTEX_SHADER, GL_FRAGMENT_SHADER = 0x8B31, 0x8B30
GL_COMPILE_STATUS, GL_LINK_STATUS = 0x8B81, 0x8B82
GL_INFO_LOG_LENGTH = 0x8B84
GL_VERSION, GL_SHADING_LANGUAGE_VERSION = 0x1F02, 0x8B8C

egl = ctypes.CDLL(ctypes.util.find_library("EGL") or "libEGL.so.1")
gl = ctypes.CDLL(ctypes.util.find_library("GLESv2") or "libGLESv2.so.2")

egl.eglGetProcAddress.restype = ctypes.c_void_p
def proc(name, restype, *argtypes):
    addr = egl.eglGetProcAddress(name.encode())
    if not addr:
        return None
    return ctypes.CFUNCTYPE(restype, *argtypes)(addr)

def make_context():
    getdisp = proc("eglGetPlatformDisplayEXT", ctypes.c_void_p,
                   ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))
    if getdisp is None:
        raise RuntimeError("eglGetPlatformDisplayEXT unavailable")
    dpy = getdisp(EGL_PLATFORM_SURFACELESS_MESA, None, None)
    if not dpy:
        raise RuntimeError("no surfaceless display")
    egl.eglInitialize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                  ctypes.POINTER(ctypes.c_int)]
    maj, mnr = ctypes.c_int(), ctypes.c_int()
    if not egl.eglInitialize(ctypes.c_void_p(dpy), ctypes.byref(maj), ctypes.byref(mnr)):
        raise RuntimeError("eglInitialize failed")
    attrs = (ctypes.c_int * 7)(EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                               EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
                               EGL_NONE, EGL_NONE, EGL_NONE)
    cfg = ctypes.c_void_p(); n = ctypes.c_int()
    egl.eglChooseConfig.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                    ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_int)]
    if not egl.eglChooseConfig(ctypes.c_void_p(dpy), attrs, ctypes.byref(cfg), 1,
                               ctypes.byref(n)) or n.value < 1:
        raise RuntimeError("no ES3 config")
    egl.eglBindAPI(EGL_OPENGL_ES_API)
    cattrs = (ctypes.c_int * 3)(EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE)
    egl.eglCreateContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.POINTER(ctypes.c_int)]
    egl.eglCreateContext.restype = ctypes.c_void_p
    ctx = egl.eglCreateContext(ctypes.c_void_p(dpy), cfg, None, cattrs)
    if not ctx:
        raise RuntimeError("eglCreateContext failed")
    egl.eglMakeCurrent.argtypes = [ctypes.c_void_p] * 4
    if not egl.eglMakeCurrent(ctypes.c_void_p(dpy), EGL_NO_SURFACE, EGL_NO_SURFACE,
                              ctypes.c_void_p(ctx)):
        raise RuntimeError("eglMakeCurrent failed")
    return dpy, ctx

def gl_string(name):
    gl.glGetString.restype = ctypes.c_char_p
    return (gl.glGetString(name) or b"?").decode()

def compile_shader(kind, src):
    sid = gl.glCreateShader(kind)
    buf = ctypes.c_char_p(src.encode())
    arr = (ctypes.c_char_p * 1)(buf)
    gl.glShaderSource(sid, 1, arr, None)
    gl.glCompileShader(sid)
    ok = ctypes.c_int()
    gl.glGetShaderiv(sid, GL_COMPILE_STATUS, ctypes.byref(ok))
    ln = ctypes.c_int()
    gl.glGetShaderiv(sid, GL_INFO_LOG_LENGTH, ctypes.byref(ln))
    log = b""
    if ln.value > 1:
        b = ctypes.create_string_buffer(ln.value)
        gl.glGetShaderInfoLog(sid, ln.value, None, b)
        log = b.value
    return bool(ok.value), log.decode(errors="ignore").strip(), sid

VERT_ES = """#version 300 es
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

# The three passes, in GLSL ES 3.00. Differences from the 330-core versions are exactly the
# ES rules: an explicit `precision highp float;` / `precision highp sampler2D;` and no more.
BIND_ES = """#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uX;
uniform sampler2D uK;
uniform int uD;
out vec4 fragOut;
void main() {
    int j = int(gl_FragCoord.x);
    float acc = 0.0;
    for (int i = 0; i < uD; ++i) {
        int idx = j - i;
        if (idx < 0) idx += uD;          // domain repetition: the circulant index wrap
        acc += texelFetch(uK, ivec2(idx, 0), 0).r * texelFetch(uX, ivec2(i, 0), 0).r;
    }
    fragOut = vec4(acc, 0.0, 0.0, 1.0);
}
"""

SCORE_ES = """#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uV;
uniform sampler2D uQ;
uniform int uD;
out vec4 fragOut;
void main() {
    int row = int(gl_FragCoord.x);
    float acc = 0.0;
    for (int i = 0; i < uD; ++i) {
        acc += texelFetch(uV, ivec2(i, row), 0).r * texelFetch(uQ, ivec2(i, 0), 0).r;
    }
    fragOut = vec4(acc, 0.0, 0.0, 1.0);
}
"""

MAXRED_ES = """#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uS;
uniform int uN;
uniform int uTile;
out vec4 fragOut;
void main() {
    int t = int(gl_FragCoord.x);
    float best = -1e30;
    for (int i = 0; i < uTile; ++i) {
        int j = t * uTile + i;
        if (j < uN) best = max(best, texelFetch(uS, ivec2(j, 0), 0).r);
    }
    fragOut = vec4(best, 0.0, 0.0, 1.0);
}
"""

if __name__ == "__main__":
    make_context()
    print("GLES:", gl_string(GL_VERSION))
    print("GLSL:", gl_string(GL_SHADING_LANGUAGE_VERSION))
    print()
    ok_v, log_v, vid = compile_shader(GL_VERTEX_SHADER, VERT_ES)
    print("vertex          : %s %s" % ("OK" if ok_v else "FAIL", log_v))
    fails = 0 if ok_v else 1
    for name, src in (("bind (circulant)", BIND_ES), ("score (matvec)", SCORE_ES),
                      ("tiled max (T4)", MAXRED_ES)):
        ok, log, fid = compile_shader(GL_FRAGMENT_SHADER, src)
        print("%-16s: %s %s" % (name, "OK" if ok else "FAIL", log))
        fails += 0 if ok else 1
        if ok and ok_v:                       # linking is a separate compiler stage; do it too
            pid = gl.glCreateProgram()
            gl.glAttachShader(pid, vid); gl.glAttachShader(pid, fid)
            gl.glLinkProgram(pid)
            st = ctypes.c_int(); gl.glGetProgramiv(pid, GL_LINK_STATUS, ctypes.byref(st))
            print("%-16s  link: %s" % ("", "OK" if st.value else "FAIL"))
            fails += 0 if st.value else 1
    print("\n%s" % ("ALL GLSL ES 3.00 SHADERS COMPILE AND LINK" if fails == 0
                    else "FAILURES: %d" % fails))
    sys.exit(1 if fails else 0)
