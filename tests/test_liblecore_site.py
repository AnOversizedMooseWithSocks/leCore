"""Static contract tests for the liblecore GitHub Pages landing page."""

from __future__ import annotations

import struct
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site" / "liblecore"


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.scripts: list[str] = []
        self.meta: dict[str, str] = {}
        self.title_parts: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "link" and values.get("href"):
            self.links.append(values["href"] or "")
        elif tag == "script" and values.get("src"):
            self.scripts.append(values["src"] or "")
        elif tag == "meta" and values.get("content"):
            key = values.get("property") or values.get("name")
            if key:
                self.meta[key] = values["content"] or ""
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)


class LiblecoreSiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (SITE / "index.html").read_text(encoding="utf-8")
        cls.parser = _PageParser()
        cls.parser.feed(cls.html)

    def test_required_static_assets_exist(self) -> None:
        expected = {
            ".nojekyll",
            "favicon.svg",
            "index.html",
            "llms.txt",
            "og.png",
            "robots.txt",
            "script.js",
            "styles.css",
        }
        self.assertTrue(expected.issubset({path.name for path in SITE.iterdir()}))

        for reference in self.parser.links + self.parser.scripts:
            parsed = urlparse(reference)
            if parsed.scheme or reference.startswith(("#", "/")):
                continue
            self.assertTrue((SITE / parsed.path).is_file(), reference)

    def test_social_metadata_is_complete(self) -> None:
        title = "".join(self.parser.title_parts).strip()
        self.assertEqual(title, "liblecore — one vector algebra, every runtime")
        self.assertEqual(self.parser.meta["og:type"], "website")
        self.assertEqual(
            self.parser.meta["og:url"],
            "https://anoversizedmoosewithsocks.github.io/leCore/",
        )
        self.assertEqual(
            self.parser.meta["og:image"],
            "https://anoversizedmoosewithsocks.github.io/leCore/og.png",
        )
        self.assertEqual(self.parser.meta["twitter:card"], "summary_large_image")

    def test_social_card_has_the_declared_dimensions(self) -> None:
        png = (SITE / "og.png").read_bytes()
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", png[16:24])
        self.assertEqual((width, height), (1200, 630))

    def test_performance_copy_matches_recorded_benchmark(self) -> None:
        benchmark = (ROOT / "native" / "liblecore" / "BENCHMARKS.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "| f32 | 1024 | 708.79 | 62.52 | 11.36x | 14.31 | 13.43 | 1.07x |",
            benchmark,
        )
        for value in ("62.52", "13.43", "1.043e−7"):
            self.assertIn(value, self.html)

    def test_published_page_has_no_local_or_placeholder_links(self) -> None:
        lowered = self.html.lower()
        self.assertNotIn("localhost", lowered)
        self.assertNotIn("127.0.0.1", lowered)
        self.assertNotIn("todo", lowered)


if __name__ == "__main__":
    unittest.main()
