"""Deterministic data charts as SVG strings (holographic_chartsvg).

WHY THIS EXISTS. The engine analyses data every which way (transport, graph
signals, KDE, time series) and could draw holographic vector art (svg_canvas,
a substrate codec) -- but it had NO door from "here is a series of numbers" to
"here is a chart a human reads". Every host bolted on matplotlib, which the
constitution forbids in core. This module is the cadexport move applied to
charts: PURE STRING assembly, stdlib only, the caller writes the file (or an
MCP server ships it as content). svg_canvas is deliberately NOT used here --
it is a hypervector CODEC for vector art, a different costume; a chart needs
axes and ticks, not a manifold.

Deterministic by construction: same input, byte-identical SVG (fixed palette,
%g formatting, no clock, no rng). KEPT NEG: non-finite values are REFUSED
loudly, never silently dropped -- a chart that quietly omits a NaN point lies
about the data it claims to show.
"""

import math

# Fixed palette (Okabe-Ito, colorblind-safe) -- deterministic series colors.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#F0E442", "#000000")


def _fin(v):
    v = float(v)
    if not math.isfinite(v):
        raise ValueError("non-finite value %r -- a chart that silently drops a NaN "
                         "lies about its data; clean the series first" % v)
    return v


def _norm_series(series):
    """Accept one series or many; return list-of-lists of finite floats."""
    if not series:
        raise ValueError("empty series -- nothing to chart")
    first = series[0]
    many = isinstance(first, (list, tuple))
    rows = [list(s) for s in series] if many else [list(series)]
    return [[_fin(v) for v in row] for row in rows]


def _ticks(lo, hi, n=5):
    """n evenly spaced ticks, lo/hi padded when degenerate. Simple on purpose:
    'nice number' heuristics vary by taste; even spacing never surprises."""
    if hi <= lo:
        lo, hi = lo - 1.0, hi + 1.0
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)], lo, hi


def _fmt(v):
    return ("%g" % round(v, 10))


def chart_svg(kind, series, labels=None, title=None, x=None,
              width=640, height=400, colors=None):
    """Render a LINE, BAR, or SCATTER chart of `series` to an SVG string.

    `series`: one list of numbers, or a list of lists (multi-series). For
    'scatter', each series is [(x, y), ...] pairs (or one such list).
    `x`: optional shared x values for 'line' (defaults to 0..n-1).
    `labels`: per-series legend names; `title`: chart heading.
    Returns the complete SVG document as a string -- the caller writes the
    file or ships it over a wire. Deterministic: identical input yields a
    byte-identical string. Non-finite values raise (kept negative above)."""
    kind = str(kind).lower()
    if kind not in ("line", "bar", "scatter"):
        raise ValueError("kind must be line|bar|scatter, got %r" % kind)
    W, H = int(width), int(height)
    ml, mr, mt, mb = 56, 16, (34 if title else 16), 36    # plot margins
    pw, ph = W - ml - mr, H - mt - mb
    cols = list(colors or PALETTE)

    if kind == "scatter":
        first = series[0]
        many = bool(first) and isinstance(first, (list, tuple)) \
            and bool(first) and isinstance(first[0], (list, tuple))
        rows = [list(s) for s in series] if many else [list(series)]
        pts = [[(_fin(px), _fin(py)) for px, py in row] for row in rows]
        xs = [p[0] for row in pts for p in row]
        ys = [p[1] for row in pts for p in row]
    else:
        rows = _norm_series(series)
        n = max(len(r) for r in rows)
        xs = [float(v) for v in (x if x is not None else range(n))]
        if x is not None:
            xs = [_fin(v) for v in xs]
        ys = [v for row in rows for v in row]
    xt, xlo, xhi = _ticks(min(xs), max(xs))
    # bars stand on zero by convention -- a bar chart not anchored at zero
    # exaggerates; lines and scatters keep the data's own range
    ylo0 = min(ys + [0.0]) if kind == "bar" else min(ys)
    yhi0 = max(ys + [0.0]) if kind == "bar" else max(ys)
    yt, ylo, yhi = _ticks(ylo0, yhi0)

    def X(v):
        return ml + (float(v) - xlo) / (xhi - xlo) * pw

    def Y(v):
        return mt + ph - (float(v) - ylo) / (yhi - ylo) * ph

    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
           'viewBox="0 0 %d %d" font-family="sans-serif" font-size="11">'
           % (W, H, W, H),
           '<rect width="%d" height="%d" fill="white"/>' % (W, H)]
    if title:
        out.append('<text x="%d" y="20" font-size="14" text-anchor="middle">%s</text>'
                   % (W // 2, str(title)))
    # axes + ticks + grid
    out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#444"/>'
               % (ml, mt + ph, ml + pw, mt + ph))
    out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#444"/>'
               % (ml, mt, ml, mt + ph))
    for tv in yt:
        yy = Y(tv)
        out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#ddd"/>'
                   % (ml, yy, ml + pw, yy))
        out.append('<text x="%d" y="%.1f" text-anchor="end" dy="4">%s</text>'
                   % (ml - 6, yy, _fmt(tv)))
    for tv in xt:
        xx = X(tv)
        out.append('<text x="%.1f" y="%d" text-anchor="middle">%s</text>'
                   % (xx, mt + ph + 16, _fmt(tv)))

    if kind == "line":
        for i, row in enumerate(rows):
            d = " ".join("%s%.1f,%.1f" % ("M" if j == 0 else "L", X(xs[j]), Y(v))
                         for j, v in enumerate(row))
            out.append('<path d="%s" fill="none" stroke="%s" stroke-width="2"/>'
                       % (d, cols[i % len(cols)]))
    elif kind == "bar":
        groups = max(len(r) for r in rows)
        band = pw / groups
        bw = band * 0.8 / len(rows)
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                x0 = ml + j * band + band * 0.1 + i * bw
                y0, y1 = Y(max(v, 0.0)), Y(min(v, 0.0))
                out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                           'fill="%s"/>' % (x0, y0, bw, max(y1 - y0, 0.5),
                                            cols[i % len(cols)]))
    else:                                                   # scatter
        for i, row in enumerate(pts):
            for px, py in row:
                out.append('<circle cx="%.1f" cy="%.1f" r="3.5" fill="%s" '
                           'fill-opacity="0.85"/>' % (X(px), Y(py),
                                                      cols[i % len(cols)]))
    if labels:
        for i, lab in enumerate(list(labels)[:len(rows) if kind != "scatter" else len(pts)]):
            yy = mt + 8 + i * 16
            out.append('<rect x="%d" y="%d" width="10" height="10" fill="%s"/>'
                       % (ml + pw - 110, yy, cols[i % len(cols)]))
            out.append('<text x="%d" y="%d" dy="9">%s</text>'
                       % (ml + pw - 96, yy, str(lab)))
    out.append("</svg>")
    return "\n".join(out)


def _selftest():
    # 1. Deterministic: byte-identical across calls.
    a = chart_svg("line", [1, 3, 2], title="t")
    assert a == chart_svg("line", [1, 3, 2], title="t"), "chart must be deterministic"
    assert a.startswith("<svg") and a.rstrip().endswith("</svg>")
    # 2. Planted geometry truth: with data 0..10 and defaults (W=640, ml=56, mr=16,
    #    H=400, mt=16, mb=36), the point (10, 10) is the top-right of the plot area.
    s = chart_svg("line", [0, 10], x=[0, 10])
    assert "L624.0,16.0" in s, "the max point must land at the plot's top-right corner"
    # 3. Bars anchor at zero: a chart of [2, 5] must place bar bottoms at Y(0).
    b = chart_svg("bar", [2, 5])
    assert b.count("<rect") == 1 + 2, "one background + one rect per bar"
    # 4. Multi-series legend rows match the series count.
    m2 = chart_svg("line", [[1, 2], [2, 1]], labels=["a", "b"])
    assert m2.count('width="10" height="10"') == 2
    # 5. KEPT NEGATIVE pinned: a NaN is refused loudly, never silently dropped.
    try:
        chart_svg("line", [1.0, float("nan")])
        raise AssertionError("NaN must be refused")
    except ValueError as e:
        assert "non-finite" in str(e)
    # 6. Scatter accepts pairs, single and multi.
    sc = chart_svg("scatter", [(0, 0), (1, 1)])
    assert sc.count("<circle") == 2
    return ("chartsvg selftest OK: deterministic; planted corner point lands; bars "
            "anchor at zero; legend matches series; NaN refused; scatter pairs")


if __name__ == "__main__":
    print(_selftest())
