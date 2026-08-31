#!/usr/bin/env python3
"""Compile the liblecore amalgamation to WebAssembly and run it under Node."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
AMALGAMATION = REPOSITORY_ROOT / "native/liblecore/amalgamation"
ABI_SYMBOLS = REPOSITORY_ROOT / "native/liblecore/ABI0_SYMBOLS.txt"

SMOKE_SOURCE = r"""
#include "lecore.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

static int close_enough(double left, double right)
{
    return fabs(left - right) <= 1e-9;
}

static int exercise_context(lecore_backend backend)
{
    const double left[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const double right[8] = {0.0, 0.25, 0.0, -0.5, 0.0, 0.75, 0.0, 1.0};
    double output[8] = {0.0};
    lecore_config_v0 config;
    lecore_context *context = NULL;
    size_t index;

    lecore_config_init_v0(&config);
    config.dimension = UINT32_C(8);
    config.backend = backend;
    if (lecore_context_create(&config, &context) != LECORE_OK || context == NULL) {
        return 1;
    }
    if (lecore_hrr_bind_f64(context, left, right, output) != LECORE_OK) {
        lecore_context_destroy(context);
        return 2;
    }
    for (index = 0; index < 8; ++index) {
        if (!close_enough(output[index], right[index])) {
            lecore_context_destroy(context);
            return 3;
        }
    }
    lecore_context_destroy(context);
    return 0;
}

int main(void)
{
    if (lecore_abi_version() != LECORE_ABI_VERSION ||
        lecore_isa_version() != LECORE_ISA_VERSION ||
        exercise_context(LECORE_BACKEND_DIRECT) != 0) {
        return 10;
    }

#if LECORE_ENABLE_RADIX2
    if ((lecore_capabilities() & LECORE_CAP_RADIX2) == 0 ||
        exercise_context(LECORE_BACKEND_RADIX2) != 0) {
        return 11;
    }
#else
    if ((lecore_capabilities() & LECORE_CAP_RADIX2) != 0) {
        return 12;
    }
#endif

#if LECORE_ENABLE_FORMAT
    {
        const uint8_t payload[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        uint8_t record[LECORE_FORMAT_HEADER_BYTES + sizeof(payload)];
        lecore_format_descriptor_v1 descriptor;
        lecore_format_descriptor_v1 decoded;
        const void *decoded_payload = NULL;
        size_t decoded_bytes = 0;
        size_t record_bytes = 0;

        lecore_format_descriptor_init_v1(&descriptor);
        descriptor.artifact_kind = LECORE_ARTIFACT_VECTOR;
        descriptor.dimension = UINT32_C(1);
        descriptor.vector_count = UINT64_C(1);
        descriptor.payload_bytes = sizeof(payload);
        if ((lecore_capabilities() & LECORE_CAP_FORMAT) == 0 ||
            lecore_format_encode_v1(&descriptor, payload, sizeof(payload),
                                    record, sizeof(record), &record_bytes) != LECORE_OK ||
            record_bytes != sizeof(record) ||
            lecore_format_decode_v1(record, record_bytes, &decoded,
                                    &decoded_payload, &decoded_bytes) != LECORE_OK ||
            decoded_bytes != sizeof(payload) ||
            ((const uint8_t *)decoded_payload)[7] != UINT8_C(8)) {
            return 13;
        }
    }
#else
    if ((lecore_capabilities() & LECORE_CAP_FORMAT) != 0) {
        return 14;
    }
#endif

    return 0;
}
"""


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def _exported_functions(format_enabled: bool) -> str:
    """Return Emscripten's exact ABI-0 export list for this feature set."""
    symbols = [
        line.strip()
        for line in ABI_SYMBOLS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not format_enabled:
        symbols = [name for name in symbols if not name.startswith("lecore_format_")]
    return json.dumps(["_main", *(f"_{name}" for name in symbols)], separators=(",", ":"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emcc", default=shutil.which("emcc"))
    parser.add_argument("--node", default=shutil.which("node"))
    arguments = parser.parse_args()
    if not arguments.emcc:
        parser.error("emcc was not found; pass --emcc")
    if not arguments.node:
        parser.error("node was not found; pass --node")

    generator = REPOSITORY_ROOT / "tools/generate_liblecore_amalgamation.py"
    _run(["python3", str(generator), "--check"])

    with tempfile.TemporaryDirectory(prefix="liblecore-wasm-") as directory:
        temporary = Path(directory)
        smoke = temporary / "smoke.c"
        smoke.write_text(SMOKE_SOURCE, encoding="utf-8")
        for label, format_enabled, radix2_enabled in (
            ("features-on", 1, 1),
            ("features-off", 0, 0),
        ):
            output = temporary / f"{label}.js"
            _run(
                [
                    arguments.emcc,
                    "-std=c11",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Wpedantic",
                    "-Wconversion",
                    "-Wshadow",
                    "-Werror",
                    "-fno-fast-math",
                    "-ffp-contract=off",
                    f"-DLECORE_ENABLE_FORMAT={format_enabled}",
                    f"-DLECORE_ENABLE_RADIX2={radix2_enabled}",
                    "-I",
                    str(AMALGAMATION),
                    str(smoke),
                    str(AMALGAMATION / "lecore.c"),
                    "-sASSERTIONS=1",
                    "-sENVIRONMENT=node",
                    "-sEXIT_RUNTIME=1",
                    "-sFILESYSTEM=0",
                    "-sSTRICT=1",
                    f"-sEXPORTED_FUNCTIONS={_exported_functions(bool(format_enabled))}",
                    "-o",
                    str(output),
                ]
            )
            _run([arguments.node, str(output)])

    print(
        "liblecore WebAssembly: feature-on/off ABI export and runtime smoke passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
