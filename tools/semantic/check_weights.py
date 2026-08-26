#!/usr/bin/env python3
"""Fail loudly if a 'weights' file is not actually a safetensors (e.g. curl saved a 404 HTML page).

WHY: without curl -f, an HTTP error saves the error PAGE as model.safetensors. read_st then reads the
page's first 8 bytes as a little-endian header length -> a garbage-huge number -> MemoryError deep in
the forward pass. Catch it here, at download time, with a message that names the real cause.
"""
import struct, sys, pathlib

p = pathlib.Path(sys.argv[1])
size = p.stat().st_size
with p.open('rb') as f:
    head8 = f.read(8)
n = struct.unpack('<Q', head8)[0]
# a real safetensors header length is positive and far smaller than the file; HTML/JSON error pages
# start with '<' or '{' which unpack to absurd lengths.
if n <= 0 or n > size or n > 100_000_000:
    sys.exit(f"  weights look wrong: header length {n} vs file size {size} bytes.\n"
             f"  The URL probably returned an error page, not the model -- check NOMIC_WEIGHTS_URL "
             f"(it must be a direct 'resolve/main/model.safetensors' link, not an HTML page).")
print(f"  weights OK: {size/1e6:.0f} MB, header {n} bytes")
