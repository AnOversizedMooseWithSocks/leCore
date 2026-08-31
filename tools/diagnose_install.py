#!/usr/bin/env python3
"""DIAGNOSE -- what does THIS model look like, before anything is installed?

Run this when an install fails on a model I cannot reproduce. It prints the
facts an install decision depends on, so a screenshot of its output is enough
to find the cause without another round trip.

    python tools/diagnose_install.py path/to/model
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(model_dir):
    import numpy as np
    # RESOLVE LIKE THE LAUNCHERS DO. diagnose.bat cds to the repo root so the
    # package imports work, which breaks any relative path typed from
    # assimilation/ -- the same bug install.py had. GALVATRON_CWD carries the
    # caller's directory and _resolve_model_dir tries it, both separator forms,
    # and work/ under the repo AND under assimilation/.
    from assimilation.galvatron import _resolve_model_dir
    model_dir = _resolve_model_dir(model_dir)
    from holographic.io_and_interop.holographic_gdnruntime import (
        load_runtime, load_weights_dir)
    from holographic.io_and_interop.holographic_adapt import infer

    rt, cfg = load_runtime(model_dir)
    w = load_weights_dir(model_dir)
    print("MODEL: %s" % model_dir)
    print("  layers %s | hidden %s | vocab %s"
          % (cfg.get("n_layers"), cfg.get("hidden"),
             np.asarray(w[next(k for k in w if "embed" in k)]).shape[0]))
    print("  dtypes: %s"
          % sorted({str(np.asarray(v).dtype) for v in w.values()}))
    arch = infer(w, tokenizer_dir=model_dir)
    print("  family %s | recurrent state %s | confidence %.2f"
          % (arch["family"], arch["has_recurrent_state"], arch["confidence"]))
    print()
    print("LAYOUT RESOLUTION (the .lecore_layout.json question):")
    print("  qkv_order   %s" % cfg.get("qkv_order", "(unambiguous -- not needed)"))
    print("  cfg is rt.cfg  %s   <- must be True, or the resolution is lost"
          % (cfg is rt.cfg))
    print()
    print("GDN GEOMETRY (what the ladder reshapes against):")
    for k in ("linear_num_key_heads", "linear_num_value_heads",
              "linear_key_head_dim", "linear_value_head_dim",
              "linear_conv_kernel_dim"):
        print("  %-26s %s" % (k, cfg.get(k)))
    print()
    print("PER-LAYER TENSOR FAMILIES (first 6 and last 2):")
    seen = []
    for L in range(int(cfg["n_layers"])):
        ks = [k for k in w if ".layers.%d." % L in k]
        fam = "linear" if any("linear_attn" in k or "in_proj" in k
                              for k in ks) else "full"
        shp = [np.asarray(w[k]).shape for k in ks if "in_proj_qkvz" in k]
        seen.append((L, fam, shp[0] if shp else None, len(ks)))
    for row in seen[:6] + [("...",)] + seen[-2:]:
        print("  %s" % (row,))
    print()
    print("BLANK-LAYER CHECK (does prepend stay bit-identical here?):")
    from holographic.io_and_interop.holographic_prepend import prepend_layers
    from holographic.io_and_interop.holographic_gdnruntime import GDNRuntime
    probe = list(range(5, 37))
    before = np.asarray(rt.forward(probe), np.float64)
    out = prepend_layers(w, cfg, n=2)
    w2, c2 = out[0], out[1]
    after = np.asarray(GDNRuntime(w2, c2).forward(probe), np.float64)
    d = float(np.max(np.abs(after - before)))
    print("  drift %.3e (relative %.3e)"
          % (d, d / (float(np.max(np.abs(before))) or 1.0)))
    nz = [(k.split(".layers.0.")[1], int((np.asarray(v) != 0).sum()),
           np.asarray(v).size)
          for k, v in sorted(w2.items()) if ".layers.0." in k]
    print("  NONZERO tensors in the blank layer (should be norms only):")
    for name, n, tot in nz:
        if n:
            print("    %-44s %d/%d" % (name[:44], n, tot))
    # THE VERDICT LINE (added after a field run read the list as a failure):
    # the operative test is the DRIFT. In hybrid families the blank layer
    # carries whole gated sublayers whose contribution is shut off by their
    # gates, so non-norm tensors can be nonzero while the layer stays exactly
    # inert -- drift 0.0e+00 means bit-identical prepend, full stop.
    if d == 0.0:
        print("  VERDICT: drift is exactly zero -- the blank layer is "
              "functionally INERT; the nonzero")
        print("  tensors above are gated off in this family and are safe. "
              "Prepend is safe here.")
    else:
        print("  VERDICT: drift is NONZERO -- the blank layer is leaking; "
              "do NOT install until this reads 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "work/original"))
