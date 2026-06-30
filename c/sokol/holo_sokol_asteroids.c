/*
    holo_sokol_asteroids.c

    A pure-C Sokol real-time renderer for holostuff's procedural-geometry side:
    build a deterministic low-poly asteroid belt on startup, upload it once to
    the GPU, then orbit a camera through the field at interactive frame rates.

    Build:
        make -C c sokol-asteroids

    Run:
        ./c/build/sokol/holo_sokol_asteroids
*/
#if defined(__APPLE__)
#define SOKOL_GLCORE
#else
#define SOKOL_GLCORE
#endif
#define SOKOL_IMPL
#include "sokol_app.h"
#include "sokol_gfx.h"
#include "sokol_glue.h"
#include "sokol_log.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASTEROID_COUNT (420)
#define ASTEROID_LAT (11)
#define ASTEROID_LON (18)
#define MAX_TRIANGLES_PER_ASTEROID (ASTEROID_LON * (2 * ASTEROID_LAT - 2))
#define MAX_VERTICES (ASTEROID_COUNT * MAX_TRIANGLES_PER_ASTEROID * 3)
#define HOLO_PI (3.14159265358979323846f)

typedef struct {
    float x, y, z;
} v3;

typedef struct {
    float m[9];
} m3;

typedef struct {
    float position[3];
    float normal[3];
    float color[3];
} vertex_t;

typedef struct {
    float mvp[16];
} vs_params_t;

static struct {
    sg_pipeline pip;
    sg_bindings bind;
    sg_pass_action pass_action;
    int vertex_count;
    double time;
} state;

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static inline v3 v3_make(float x, float y, float z) {
    v3 r = { x, y, z };
    return r;
}

static inline v3 v3_add(v3 a, v3 b) {
    return v3_make(a.x + b.x, a.y + b.y, a.z + b.z);
}

static inline v3 v3_sub(v3 a, v3 b) {
    return v3_make(a.x - b.x, a.y - b.y, a.z - b.z);
}

static inline v3 v3_scale(v3 a, float s) {
    return v3_make(a.x * s, a.y * s, a.z * s);
}

static inline float v3_dot(v3 a, v3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline v3 v3_cross(v3 a, v3 b) {
    return v3_make(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

static inline v3 v3_norm(v3 a) {
    const float n = sqrtf(v3_dot(a, a));
    return n > 1.0e-8f ? v3_scale(a, 1.0f / n) : v3_make(0.0f, 1.0f, 0.0f);
}

static uint32_t rng_u32(uint32_t* s) {
    *s = (*s * 1664525u) + 1013904223u;
    return *s;
}

static float rng01(uint32_t* s) {
    return (float)((rng_u32(s) >> 8) & 0x00FFFFFFu) / 16777215.0f;
}

static float rng_range(uint32_t* s, float lo, float hi) {
    return lo + (hi - lo) * rng01(s);
}

static float rng_normal(uint32_t* s) {
    float x = 0.0f;
    for (int i = 0; i < 6; i++) {
        x += rng01(s);
    }
    return x - 3.0f;
}

static m3 m3_rotation(v3 axis, float angle) {
    axis = v3_norm(axis);
    const float c = cosf(angle);
    const float sn = sinf(angle);
    const float C = 1.0f - c;
    const float x = axis.x;
    const float y = axis.y;
    const float z = axis.z;
    m3 r = {{
        c + x * x * C,       x * y * C - z * sn,  x * z * C + y * sn,
        y * x * C + z * sn,  c + y * y * C,       y * z * C - x * sn,
        z * x * C - y * sn,  z * y * C + x * sn,  c + z * z * C
    }};
    return r;
}

static v3 m3_mul_v3(m3 m, v3 v) {
    return v3_make(
        m.m[0] * v.x + m.m[1] * v.y + m.m[2] * v.z,
        m.m[3] * v.x + m.m[4] * v.y + m.m[5] * v.z,
        m.m[6] * v.x + m.m[7] * v.y + m.m[8] * v.z);
}

static float hash31(float x, float y, float z, float seed) {
    const float h = sinf(x * 127.1f + y * 311.7f + z * 74.7f + seed * 19.19f) * 43758.5453f;
    return h - floorf(h);
}

static float asteroid_radius(v3 d, uint32_t seed) {
    const float s = (float)(seed & 1023u);
    float n = 0.0f;
    float amp = 0.18f;
    float freq = 2.0f;
    for (int i = 0; i < 4; i++) {
        const float h = hash31(d.x * freq, d.y * freq, d.z * freq, s + (float)i * 37.0f);
        n += amp * (2.0f * h - 1.0f);
        amp *= 0.52f;
        freq *= 1.9f;
    }
    const float ridge = 0.08f * sinf(7.0f * d.x + 3.0f * d.y - 5.0f * d.z + s * 0.017f);
    return clampf(1.0f + n + ridge, 0.58f, 1.48f);
}

static void mat4_identity(float m[16]) {
    memset(m, 0, sizeof(float) * 16);
    m[0] = 1.0f;
    m[5] = 1.0f;
    m[10] = 1.0f;
    m[15] = 1.0f;
}

static void mat4_mul(float out[16], const float a[16], const float b[16]) {
    float r[16];
    for (int c = 0; c < 4; c++) {
        for (int row = 0; row < 4; row++) {
            r[c * 4 + row] =
                a[0 * 4 + row] * b[c * 4 + 0] +
                a[1 * 4 + row] * b[c * 4 + 1] +
                a[2 * 4 + row] * b[c * 4 + 2] +
                a[3 * 4 + row] * b[c * 4 + 3];
        }
    }
    memcpy(out, r, sizeof(r));
}

static void mat4_perspective(float out[16], float fovy_rad, float aspect, float znear, float zfar) {
    mat4_identity(out);
    const float f = 1.0f / tanf(fovy_rad * 0.5f);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = (zfar + znear) / (znear - zfar);
    out[11] = -1.0f;
    out[14] = (2.0f * zfar * znear) / (znear - zfar);
    out[15] = 0.0f;
}

static void mat4_look_at(float out[16], v3 eye, v3 center, v3 up_hint) {
    const v3 f = v3_norm(v3_sub(center, eye));
    const v3 s = v3_norm(v3_cross(f, up_hint));
    const v3 u = v3_cross(s, f);
    mat4_identity(out);
    out[0] = s.x;
    out[1] = u.x;
    out[2] = -f.x;
    out[4] = s.y;
    out[5] = u.y;
    out[6] = -f.y;
    out[8] = s.z;
    out[9] = u.z;
    out[10] = -f.z;
    out[12] = -v3_dot(s, eye);
    out[13] = -v3_dot(u, eye);
    out[14] = v3_dot(f, eye);
}

static void append_tri(vertex_t* vertices, int* vertex_count, v3 a, v3 b, v3 c, v3 color) {
    const v3 n = v3_norm(v3_cross(v3_sub(b, a), v3_sub(c, a)));
    vertex_t tri[3] = {
        {{a.x, a.y, a.z}, {n.x, n.y, n.z}, {color.x, color.y, color.z}},
        {{b.x, b.y, b.z}, {n.x, n.y, n.z}, {color.x, color.y, color.z}},
        {{c.x, c.y, c.z}, {n.x, n.y, n.z}, {color.x, color.y, color.z}},
    };
    memcpy(&vertices[*vertex_count], tri, sizeof(tri));
    *vertex_count += 3;
}

static void build_asteroid(vertex_t* vertices, int* vertex_count, uint32_t seed, v3 pos, float size, m3 rot, v3 color, v3 ellipsoid) {
    v3 grid[(ASTEROID_LAT + 1) * ASTEROID_LON];
    for (int i = 0; i <= ASTEROID_LAT; i++) {
        const float theta = HOLO_PI * (float)i / (float)ASTEROID_LAT;
        const float st = sinf(theta);
        const float ct = cosf(theta);
        for (int j = 0; j < ASTEROID_LON; j++) {
            const float phi = 2.0f * HOLO_PI * (float)j / (float)ASTEROID_LON;
            const v3 d = v3_make(st * cosf(phi), st * sinf(phi), ct);
            const float r = asteroid_radius(d, seed);
            v3 p = v3_make(d.x * ellipsoid.x, d.y * ellipsoid.y, d.z * ellipsoid.z);
            p = m3_mul_v3(rot, v3_scale(p, r * size));
            grid[i * ASTEROID_LON + j] = v3_add(pos, p);
        }
    }
    for (int i = 0; i < ASTEROID_LAT; i++) {
        for (int j = 0; j < ASTEROID_LON; j++) {
            const int jn = (j + 1) % ASTEROID_LON;
            const v3 a = grid[i * ASTEROID_LON + j];
            const v3 b = grid[i * ASTEROID_LON + jn];
            const v3 c = grid[(i + 1) * ASTEROID_LON + jn];
            const v3 d = grid[(i + 1) * ASTEROID_LON + j];
            const float tint = 0.86f + 0.22f * hash31(pos.x + (float)i, pos.y, pos.z + (float)j, (float)seed);
            const v3 face_color = v3_scale(color, tint);
            if (i == 0) {
                append_tri(vertices, vertex_count, a, c, d, face_color);
            } else if (i == ASTEROID_LAT - 1) {
                append_tri(vertices, vertex_count, a, b, d, face_color);
            } else {
                append_tri(vertices, vertex_count, a, b, c, face_color);
                append_tri(vertices, vertex_count, a, c, d, face_color);
            }
        }
    }
}

static vertex_t* build_belt_mesh(int* out_vertex_count) {
    vertex_t* vertices = (vertex_t*)calloc((size_t)MAX_VERTICES, sizeof(vertex_t));
    if (!vertices) {
        return NULL;
    }
    uint32_t rng = 0xC0DEC0DEu;
    int vertex_count = 0;
    for (int i = 0; i < ASTEROID_COUNT; i++) {
        const float angle = rng_range(&rng, 0.0f, 2.0f * HOLO_PI);
        const float radius = clampf(9.0f + rng_normal(&rng) * 1.55f, 5.7f, 13.5f);
        const float y = rng_normal(&rng) * (0.24f + 0.035f * fabsf(radius - 9.0f));
        const v3 pos = v3_make(radius * cosf(angle), y, 0.64f * radius * sinf(angle));
        float size = expf(-1.75f + 0.48f * rng_normal(&rng));
        if (rng01(&rng) < 0.07f) {
            size *= rng_range(&rng, 1.5f, 2.65f);
        }
        const v3 axis = v3_make(rng_normal(&rng), rng_normal(&rng), rng_normal(&rng));
        const m3 rot = m3_rotation(axis, rng_range(&rng, 0.0f, 2.0f * HOLO_PI));
        const v3 ellipsoid = v3_make(
            rng_range(&rng, 0.78f, 1.26f),
            rng_range(&rng, 0.76f, 1.20f),
            rng_range(&rng, 0.72f, 1.16f));
        const float warm = rng_range(&rng, 0.78f, 1.18f);
        v3 color = v3_make(
            clampf(0.48f * warm + rng_range(&rng, -0.04f, 0.05f), 0.18f, 0.86f),
            clampf(0.42f * warm + rng_range(&rng, -0.04f, 0.05f), 0.18f, 0.82f),
            clampf(0.35f * warm + rng_range(&rng, -0.04f, 0.05f), 0.18f, 0.78f));
        build_asteroid(vertices, &vertex_count, rng_u32(&rng), pos, size, rot, color, ellipsoid);
    }
    *out_vertex_count = vertex_count;
    return vertices;
}

static sg_shader make_shader(void) {
    const char* vs_src =
        "#version 330\n"
        "uniform mat4 mvp;\n"
        "in vec3 position;\n"
        "in vec3 normal;\n"
        "in vec3 color0;\n"
        "out vec3 v_color;\n"
        "out float v_light;\n"
        "void main() {\n"
        "  vec3 n = normalize(normal);\n"
        "  vec3 key = normalize(vec3(0.55, 0.48, -0.68));\n"
        "  vec3 fill = normalize(vec3(-0.35, 0.20, 0.55));\n"
        "  float diffuse = max(dot(n, key), 0.0);\n"
        "  float bounce = max(dot(n, fill), 0.0);\n"
        "  float rim = pow(1.0 - abs(n.z), 2.0);\n"
        "  v_light = 0.18 + 0.95 * diffuse + 0.25 * bounce + 0.20 * rim;\n"
        "  v_color = color0;\n"
        "  gl_Position = mvp * vec4(position, 1.0);\n"
        "}\n";
    const char* fs_src =
        "#version 330\n"
        "in vec3 v_color;\n"
        "in float v_light;\n"
        "out vec4 frag_color;\n"
        "void main() {\n"
        "  vec3 fog = vec3(0.025, 0.035, 0.060);\n"
        "  vec3 c = mix(fog, v_color * v_light, 0.92);\n"
        "  frag_color = vec4(c, 1.0);\n"
        "}\n";
    return sg_make_shader(&(sg_shader_desc){
        .vertex_func.source = vs_src,
        .fragment_func.source = fs_src,
        .attrs = {
            [0] = { .glsl_name = "position" },
            [1] = { .glsl_name = "normal" },
            [2] = { .glsl_name = "color0" },
        },
        .uniform_blocks = {
            [0] = {
                .stage = SG_SHADERSTAGE_VERTEX,
                .size = sizeof(vs_params_t),
                .layout = SG_UNIFORMLAYOUT_NATIVE,
                .glsl_uniforms = {
                    [0] = { .type = SG_UNIFORMTYPE_MAT4, .glsl_name = "mvp" },
                },
            },
        },
        .label = "asteroid-belt-shader",
    });
}

static void init(void) {
    sg_setup(&(sg_desc){
        .environment = sglue_environment(),
        .logger.func = slog_func,
    });

    int vertex_count = 0;
    vertex_t* vertices = build_belt_mesh(&vertex_count);
    if (!vertices) {
        fprintf(stderr, "failed to allocate asteroid belt mesh\n");
        sapp_request_quit();
        return;
    }
    state.vertex_count = vertex_count;
    state.bind.vertex_buffers[0] = sg_make_buffer(&(sg_buffer_desc){
        .data = { .ptr = vertices, .size = (size_t)vertex_count * sizeof(vertex_t) },
        .label = "asteroid-belt-vertices",
    });
    free(vertices);

    sg_shader shd = make_shader();
    state.pip = sg_make_pipeline(&(sg_pipeline_desc){
        .shader = shd,
        .layout = {
            .attrs = {
                [0] = { .format = SG_VERTEXFORMAT_FLOAT3 },
                [1] = { .format = SG_VERTEXFORMAT_FLOAT3 },
                [2] = { .format = SG_VERTEXFORMAT_FLOAT3 },
            },
        },
        .depth = {
            .compare = SG_COMPAREFUNC_LESS_EQUAL,
            .write_enabled = true,
        },
        .cull_mode = SG_CULLMODE_BACK,
        .face_winding = SG_FACEWINDING_CCW,
        .label = "asteroid-belt-pipeline",
    });

    state.pass_action = (sg_pass_action){
        .colors[0] = {
            .load_action = SG_LOADACTION_CLEAR,
            .clear_value = { 0.004f, 0.006f, 0.012f, 1.0f },
        },
    };
    printf("holo_sokol_asteroids: uploaded %d vertices, %.1f MB\n",
        state.vertex_count,
        ((double)state.vertex_count * (double)sizeof(vertex_t)) / (1024.0 * 1024.0));
}

static void frame(void) {
    state.time += sapp_frame_duration();
    const float w = sapp_widthf();
    const float h = sapp_heightf();
    const float aspect = w / (h > 1.0f ? h : 1.0f);
    const float t = (float)state.time;

    const v3 eye = v3_make(14.0f * sinf(t * 0.075f), 4.2f + 1.0f * sinf(t * 0.11f), 14.0f * cosf(t * 0.075f));
    const v3 target = v3_make(0.0f, 0.0f, 0.0f);

    float proj[16];
    float view[16];
    vs_params_t vs;
    mat4_perspective(proj, 55.0f * HOLO_PI / 180.0f, aspect, 0.1f, 80.0f);
    mat4_look_at(view, eye, target, v3_make(0.0f, 1.0f, 0.0f));
    mat4_mul(vs.mvp, proj, view);

    sg_begin_pass(&(sg_pass){
        .action = state.pass_action,
        .swapchain = sglue_swapchain(),
    });
    sg_apply_pipeline(state.pip);
    sg_apply_bindings(&state.bind);
    sg_apply_uniforms(0, SG_RANGE_REF(vs));
    sg_draw(0, state.vertex_count, 1);
    sg_end_pass();
    sg_commit();
}

static void cleanup(void) {
    sg_shutdown();
}

static void event(const sapp_event* ev) {
    if ((ev->type == SAPP_EVENTTYPE_KEY_DOWN) && (ev->key_code == SAPP_KEYCODE_ESCAPE)) {
        sapp_request_quit();
    }
}

sapp_desc sokol_main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    return (sapp_desc){
        .init_cb = init,
        .frame_cb = frame,
        .cleanup_cb = cleanup,
        .event_cb = event,
        .width = 1280,
        .height = 800,
        .sample_count = 4,
        .window_title = "holostuff Sokol asteroid belt",
        .logger.func = slog_func,
    };
}
