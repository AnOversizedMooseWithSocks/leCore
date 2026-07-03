# Sokol real-time renderer

`holo_sokol_asteroids.c` is a standalone pure-C renderer demo for holostuff's
procedural geometry path. It uses vendored Sokol headers from `vendor/sokol`,
builds a deterministic low-poly asteroid belt on startup, uploads it to one GPU
vertex buffer, and renders it with a depth-tested orbit camera.

Build:

```sh
make sokol-asteroids
```

Run:

```sh
make sokol-run
```

On macOS the target compiles the C source as Objective-C because Sokol's
`sokol_app.h` Cocoa backend is implemented with Objective-C internally. The
renderer code and generated geometry are plain C.
