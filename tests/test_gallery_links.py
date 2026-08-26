"""Every image GALLERY.md points at must actually exist.

Broken image links rot silently: the doc renders, the picture is a broken icon, and nobody notices until someone
reads the page. Two ways it happened here:
  * three benchmark figures were linked as `gallery/bench_*.png` but live in `benchmarks/`;
  * `gallery/render_cloud.png` was linked and had a generator in tools/make_gallery.py, but was never produced.

This test is cheap (it only stats files) and pins both.
"""
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GALLERY = os.path.join(REPO, "GALLERY.md")

_IMAGE_LINK = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def _links():
    with open(GALLERY, encoding="utf-8", errors="ignore") as fh:
        return _IMAGE_LINK.findall(fh.read())


def test_gallery_has_images():
    assert len(_links()) > 10, "GALLERY.md should link a gallery's worth of images"


def test_every_gallery_image_link_resolves():
    """Links are relative to the repo root (where GALLERY.md lives)."""
    missing = [ln for ln in _links()
               if not ln.startswith(("http://", "https://")) and not os.path.exists(os.path.join(REPO, ln))]
    assert not missing, "GALLERY.md links images that don't exist:\n" + "\n".join("  " + m for m in missing)


def test_gallery_images_are_not_empty():
    """A zero-byte or truncated PNG renders as a broken image just like a missing one."""
    tiny = []
    for ln in _links():
        if ln.startswith(("http://", "https://")):
            continue
        path = os.path.join(REPO, ln)
        if os.path.exists(path) and os.path.getsize(path) < 1024:
            tiny.append("%s (%d bytes)" % (ln, os.path.getsize(path)))
    assert not tiny, "GALLERY.md images that look empty/truncated:\n" + "\n".join("  " + t for t in tiny)
