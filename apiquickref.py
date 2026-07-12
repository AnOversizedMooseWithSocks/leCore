#!/usr/bin/env python3
"""
apiquickref.py -- generate a SCANNABLE quick reference for the app-building surface of leCore.

WHY THIS EXISTS (and how it differs from docgen.py). `docgen.py` writes REFERENCE.md: every one of the ~280
engine modules, exhaustively. That is the right tool for "read the whole engine", but the wrong tool for the
everyday question "what can RenderSession / Mesh / the camera actually do?" -- you don't want to scroll 280
modules to find it. This script emits ONE LINE per public class/function (signature + the first sentence of its
docstring) for a small CURATED set of modules -- the practical surface an app builder touches. It is a page you
can scan, not an essay.

OLD-SCHOOL AND DEPENDENCY-FREE, same as docgen: standard library only (`ast`, no importing the modules), so it
is safe and fast and can run in CI to keep API_QUICKREF.md current on its own.

    Run it:   python apiquickref.py
    Output:   API_QUICKREF.md
"""

import ast
import os
from datetime import date

# ----------------------------------------------------------------------------------------------------------
# THE CURATED SURFACE. Edit this list to change what the quick reference covers. Grouped by the job a builder
# is doing, in the order they meet it: product wedge -> author a scene -> model geometry -> aim a camera -> render -> ship.
# Kept deliberately SHORT -- the point is a page you can scan, not a full index (that is REFERENCE.md).
# ----------------------------------------------------------------------------------------------------------

CURATED = [
    ("Product wedge", ["holographic_product", "holographic_x402_api"]),
    ("Scene authoring", ["holographic_scene_doc", "holographic_modifier"]),
    ("Geometry / SDF",  ["holographic_sdf", "holographic_sdfscene", "holographic_mesh"]),
    ("Transforms",      ["holographic_transform"]),
    ("Camera",          ["holographic_camera"]),
    ("Rendering",       ["holographic_render", "holographic_pipeline", "holographic_session", "holographic_cancel"]),
    ("Export / LOD",    ["holographic_lod", "holographic_gltf"]),
]


def signature(node):
    """Build a readable `name(arg, arg=default, ...)` string from a function's AST node. Defaults are shown as    their source text where simple (numbers, strings, names) and as '...' otherwise, so the line stays short."""
    a = node.args
    params = []
    posonly = getattr(a, "posonlyargs", [])
    all_pos = posonly + a.args
    # line defaults up with the trailing positional args they belong to.
    defaults = list(a.defaults)
    n_no_default = len(all_pos) - len(defaults)
    for i, arg in enumerate(all_pos):
        if i >= n_no_default:
            params.append("%s=%s" % (arg.arg, _default_src(defaults[i - n_no_default])))
        else:
            params.append(arg.arg)
    if a.vararg:
        params.append("*" + a.vararg.arg)
    for i, arg in enumerate(a.kwonlyargs):
        d = a.kw_defaults[i]
        params.append("%s=%s" % (arg.arg, _default_src(d)) if d is not None else arg.arg)
    if a.kwarg:
        params.append("**" + a.kwarg.arg)
    return "%s(%s)" % (node.name, ", ".join(params))


def _default_src(node):
    """Show simple literals/names verbatim; anything else as '...' to keep signatures short and readable."""
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, (ast.UnaryOp, ast.Tuple, ast.List)):
        try:
            return ast.unparse(node)                 # short compound defaults like (0,1,0) read fine
        except Exception:
            return "..."
    return "..."


def first_sentence(node):
    """The first sentence of a node's docstring (up to the first period or newline), trimmed. '' if none."""
    doc = ast.get_docstring(node) or ""
    doc = " ".join(doc.split())                      # collapse whitespace/newlines
    if not doc:
        return ""
    for stop in (". ", ".\n"):
        if stop in doc:
            return doc.split(stop, 1)[0].strip() + "."
    return doc if doc.endswith(".") else doc + ("." if doc else "")


def module_entries(path):
    """Return (module_doc_first_sentence, [lines]) for one module: one line per PUBLIC function/class, and for
    classes, their public methods indented under them. Parses with `ast` -- no import, so nothing runs."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    mod_doc = first_sentence(tree)
    lines = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            lines.append("- `%s` -- %s" % (signature(node), first_sentence(node)))
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            lines.append("- **class `%s`** -- %s" % (node.name, first_sentence(node)))
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and not sub.name.startswith("_"):
                    lines.append("    - `%s` -- %s" % (signature(sub), first_sentence(sub)))
    return mod_doc, lines


def _resolve_module_path(root, mod):
    """Find a curated module's actual file, whether it's flat at the repo root (old layout) or nested under
    holographic/<family>/ (current layout)."""
    flat = os.path.join(root, mod + ".py")
    if os.path.exists(flat):
        return flat
    holo_root = os.path.join(root, "holographic")
    if os.path.isdir(holo_root):
        for dirpath, _, filenames in os.walk(holo_root):
            if mod + ".py" in filenames:
                return os.path.join(dirpath, mod + ".py")
    return flat  # doesn't exist either way; the not-found branch below handles it


def generate(root="."):
    """Write API_QUICKREF.md for the CURATED surface. Returns the output path."""
    out = ["# leCore API Quick Reference",
           "",
           "*A scannable, one-line-per-symbol map of the app-building surface -- auto-generated by "
           "`apiquickref.py` on %s. For the full engine (every module), see REFERENCE.md.*" % date.today().isoformat(),
           ""]
    for section, modules in CURATED:
        out.append("## %s" % section)
        out.append("")
        for mod in modules:
            path = _resolve_module_path(root, mod)
            if not os.path.exists(path):
                out.append("### `%s` -- *(module not found)*" % mod)
                out.append("")
                continue
            mod_doc, lines = module_entries(path)
            out.append("### `%s`" % mod)
            if mod_doc:
                out.append("*%s*" % mod_doc)
            out.append("")
            out.extend(lines if lines else ["- *(no public API)*"])
            out.append("")
    text = "\n".join(out).rstrip() + "\n"
    dest = os.path.join(root, "API_QUICKREF.md")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    return dest


if __name__ == "__main__":
    p = generate()
    print("wrote", p)
