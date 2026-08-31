#!/usr/bin/env python3
"""Check liblecore amalgamation drift and standalone compiler consumption."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPOSITORY_ROOT / "tools/generate_liblecore_amalgamation.py"
AMALGAMATION = REPOSITORY_ROOT / "native/liblecore/amalgamation"
SYMBOL_CHECKER = REPOSITORY_ROOT / "tests/native/check_liblecore_symbols.py"

C_SMOKE = r"""
#include "lecore.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static int close_enough(double left, double right)
{
    return fabs(left - right) <= 1e-9;
}

int main(void)
{
    const double left[4] = {1.0, 2.0, 0.0, -1.0};
    const double right[4] = {2.0, 0.0, 1.0, 0.0};
    const double expected[4] = {2.0, 3.0, 1.0, 0.0};
    double output[4];
    lecore_config_v0 config;
    lecore_context *context = NULL;
    lecore_status status;
    size_t index;

    if (lecore_abi_version() != 0 || lecore_isa_version() != @ISA_VERSION@ ||
        strcmp(lecore_version_string(), "@VERSION@") != 0) {
        return 1;
    }

    lecore_config_init_v0(&config);
    config.dimension = 4;
    config.backend = LECORE_BACKEND_RADIX2;
    status = lecore_context_create(&config, &context);
#if LECORE_ENABLE_RADIX2
    if (status != LECORE_OK || context == NULL ||
        lecore_context_backend(context) != LECORE_BACKEND_RADIX2 ||
        (lecore_capabilities() & LECORE_CAP_RADIX2) == 0) {
        return 2;
    }
#else
    if (status != LECORE_EUNSUPPORTED || context != NULL ||
        (lecore_capabilities() & LECORE_CAP_RADIX2) != 0) {
        return 3;
    }
    config.backend = LECORE_BACKEND_DIRECT;
    if (lecore_context_create(&config, &context) != LECORE_OK) {
        return 4;
    }
#endif

    if (lecore_hrr_bind_f64(context, left, right, output) != LECORE_OK) {
        return 5;
    }
    for (index = 0; index < 4; ++index) {
        if (!close_enough(output[index], expected[index])) {
            return 6;
        }
    }
    lecore_context_destroy(context);

#if LECORE_ENABLE_FORMAT
    {
        lecore_format_descriptor_v1 descriptor;
        lecore_format_descriptor_init_v1(&descriptor);
        if ((lecore_capabilities() & LECORE_CAP_FORMAT) == 0 ||
            descriptor.struct_size != sizeof(descriptor)) {
            return 7;
        }
    }
#else
    if ((lecore_capabilities() & LECORE_CAP_FORMAT) != 0) {
        return 8;
    }
#endif
    return 0;
}
"""

CXX_SMOKE = r"""
#include "lecore.h"

#include <cstdint>
#include <cstring>

static_assert(LECORE_ABI_VERSION == 0, "unexpected preview ABI");

int main()
{
    lecore_config_v0 config{};
    lecore_context *context = nullptr;
    lecore_config_init_v0(&config);
    config.dimension = 1;
    if (lecore_context_create(&config, &context) != LECORE_OK) {
        return 1;
    }
    const bool valid = lecore_context_dimension(context) == UINT32_C(1) &&
        std::strcmp(lecore_version_string(), "@VERSION@") == 0;
    lecore_context_destroy(context);
    return valid ? 0 : 2;
}
"""


def command_from_environment(name: str, fallback: str) -> list[str]:
    return shlex.split(os.environ.get(name, fallback))


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def strict_c_flags() -> list[str]:
    return [
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Wconversion",
        "-Wshadow",
        "-Werror",
        "-fno-fast-math",
        "-ffp-contract=off",
    ]


def main() -> int:
    cc = command_from_environment("CC", "cc")
    cxx = command_from_environment("CXX", "c++")
    run([sys.executable, str(GENERATOR), "--check"])
    version = (REPOSITORY_ROOT / "native/liblecore/VERSION").read_text(
        encoding="utf-8"
    ).strip()
    isa_version = (REPOSITORY_ROOT / "native/liblecore/ISA_VERSION").read_text(
        encoding="utf-8"
    ).strip()

    with tempfile.TemporaryDirectory(prefix="liblecore-amalgamation-") as directory:
        temporary = Path(directory)
        c_smoke = temporary / "smoke.c"
        cxx_smoke = temporary / "smoke.cpp"
        c_smoke.write_text(
            C_SMOKE.replace("@VERSION@", version).replace(
                "@ISA_VERSION@", isa_version
            ),
            encoding="utf-8",
        )
        cxx_smoke.write_text(
            CXX_SMOKE.replace("@VERSION@", version), encoding="utf-8"
        )

        for label, format_enabled, radix2_enabled in (
            ("features-on", 1, 1),
            ("features-off", 0, 0),
        ):
            defines = [
                f"-DLECORE_ENABLE_FORMAT={format_enabled}",
                f"-DLECORE_ENABLE_RADIX2={radix2_enabled}",
            ]
            implementation = temporary / f"{label}.o"
            executable = temporary / label
            run(
                cc
                + strict_c_flags()
                + defines
                + [
                    "-I",
                    str(AMALGAMATION),
                    "-c",
                    str(AMALGAMATION / "lecore.c"),
                    "-o",
                    str(implementation),
                ]
            )
            run(
                [
                    sys.executable,
                    str(SYMBOL_CHECKER),
                    "--library",
                    str(implementation),
                    "--format",
                    "on" if format_enabled else "off",
                ]
            )
            run(
                cc
                + strict_c_flags()
                + defines
                + [
                    "-I",
                    str(AMALGAMATION),
                    str(c_smoke),
                    str(implementation),
                    "-lm",
                    "-o",
                    str(executable),
                ]
            )
            run([str(executable)])

        for label, format_enabled, radix2_enabled in (
            ("features-on", 1, 1),
            ("features-off", 0, 0),
        ):
            cxx_executable = temporary / f"cxx-header-{label}"
            run(
                cxx
                + [
                    "-std=c++17",
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
                    str(cxx_smoke),
                    str(temporary / f"{label}.o"),
                    "-lm",
                    "-o",
                    str(cxx_executable),
                ]
            )
            run([str(cxx_executable)])

    print("liblecore amalgamation: drift and standalone consumers passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
