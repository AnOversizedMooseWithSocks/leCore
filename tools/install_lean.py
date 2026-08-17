#!/usr/bin/env python3
"""Opt-in Lean 4 installer -- NEVER a dependency, always a choice.

WHY THIS FILE EXISTS: the Lean 4 toolchain is ~265 MB compressed / ~1.3 GB installed --
larger than the entire leCore codebase. The house posture (pinned in NOTES) is that leCore's
logic stack is complete WITHOUT it: the Horn kernel, the independent checker, the Lean-source
EMITTER (emitting needs no binary), the fuzz oracle's non-Lean stages, and the
verified-knowledge memory all run on NumPy + stdlib. The binary buys exactly one thing: the
"lean_verified" provenance tier -- an EXTERNAL kernel's verdict. When a session wants that
tier, this script fetches it; when it doesn't, nothing here runs.

Discipline:
  * VERSION AND CHECKSUM PINNED. A verifier you download unverified is a joke at your own
    expense; the sha256 below was taken from a release this repo's exports were actually
    verified against (2026-08-16 session, 793/793 external typechecks).
  * Installs into a LOCAL prefix (default ~/.lecore/lean4), touches no system paths, prints
    the PATH line instead of editing shell rc files -- reversible by deleting one directory.
  * stdlib only (urllib, tarfile, hashlib, shutil); the zstd stream needs the `zstandard`
    pip package OR a system `zstd` binary -- whichever is present; says so honestly if neither.
  * Idempotent: an existing verified install is reported, not re-downloaded.

Usage:
    python3 tools/install_lean.py            # install if absent, report path
    python3 tools/install_lean.py --status   # report only, download nothing
    python3 tools/install_lean.py --remove   # delete the local install

leCore-side: mind.lean_status() reports the current tier; mind.lean_verify() and
verify='external' paths simply find `lean` on PATH -- export PATH per this script's output.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

# The pin. Bump BOTH lines together, and only after a session has re-run the fuzz oracle's
# external stage against the new version (793/793 or better, failures dispositioned).
LEAN_VERSION = "4.15.0"
LEAN_SHA256 = "af71a2569a9f68337de2434829b3008cd8e32c436e9cb6bd8c84a2c2ba3585c9"
LEAN_URL = ("https://github.com/leanprover/lean4/releases/download/"
            "v%s/lean-%s-linux.tar.zst" % (LEAN_VERSION, LEAN_VERSION))

PREFIX = os.path.expanduser(os.environ.get("LECORE_LEAN_PREFIX", "~/.lecore/lean4"))


def status():
    """Report the install tier without downloading anything."""
    on_path = shutil.which("lean")
    local_bin = os.path.join(PREFIX, "lean-%s-linux" % LEAN_VERSION, "bin", "lean")
    local = local_bin if os.path.exists(local_bin) else None
    ver = None
    exe = on_path or local
    if exe:
        try:
            ver = subprocess.run([exe, "--version"], capture_output=True, text=True,
                                 timeout=20).stdout.strip()
        except Exception:
            ver = "(present but not runnable)"
    return {"on_path": on_path, "local_install": local, "version": ver,
            "pinned_version": LEAN_VERSION,
            "path_hint": None if on_path else (
                "export PATH=%s:$PATH" % os.path.dirname(local) if local else None)}


def _decompress_zst(src, dst_dir):
    """zstandard pip package if importable, else a system zstd binary; honest failure text
    otherwise -- this script adds no hard dependency of its own."""
    try:
        import zstandard  # noqa: WPS433 -- optional, checked at use
        with open(src, "rb") as f:
            reader = zstandard.ZstdDecompressor().stream_reader(f)
            with tarfile.open(fileobj=reader, mode="r|") as t:
                t.extractall(dst_dir)
        return True
    except ImportError:
        pass
    if shutil.which("zstd"):
        tar = src[:-4]
        subprocess.run(["zstd", "-d", "-f", src, "-o", tar], check=True)
        with tarfile.open(tar) as t:
            t.extractall(dst_dir)
        os.remove(tar)
        return True
    print("Need either `pip install zstandard` or a system `zstd` binary to unpack "
          "the release. Neither found; nothing was installed.", file=sys.stderr)
    return False


def install():
    st = status()
    if st["version"]:
        print("Lean already available: %s" % st["version"])
        if st["path_hint"]:
            print(st["path_hint"])
        return 0
    os.makedirs(PREFIX, exist_ok=True)
    archive = os.path.join(PREFIX, "lean.tar.zst")
    print("Downloading Lean %s (~265 MB -- this is exactly why it is opt-in)..."
          % LEAN_VERSION)
    urllib.request.urlretrieve(LEAN_URL, archive)
    digest = hashlib.sha256()
    with open(archive, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != LEAN_SHA256:
        os.remove(archive)
        print("CHECKSUM MISMATCH -- refusing to install an unverified verifier. "
              "Expected %s, got %s." % (LEAN_SHA256, digest.hexdigest()), file=sys.stderr)
        return 1
    print("Checksum verified. Unpacking...")
    if not _decompress_zst(archive, PREFIX):
        return 1
    os.remove(archive)
    st = status()
    print("Installed: %s" % st["version"])
    print(st["path_hint"])
    return 0


def remove():
    if os.path.isdir(PREFIX):
        shutil.rmtree(PREFIX)
        print("Removed %s -- leCore's logic stack is unaffected; the lean_verified "
              "provenance tier is simply unavailable until reinstall." % PREFIX)
    else:
        print("No local install at %s." % PREFIX)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--remove", action="store_true")
    a = ap.parse_args()
    if a.status:
        for k, v in status().items():
            print("%s: %s" % (k, v))
        sys.exit(0)
    sys.exit(remove() if a.remove else install())
