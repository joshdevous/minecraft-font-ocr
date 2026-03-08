"""Tests for minecraft_ocr.pipeline

All tests use the bundled vanilla atlas and the synthetic renderer as the
ground-truth image source.  No real screenshots or Minecraft installation needed.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from minecraft_ocr.atlas import FontAtlas, CELL_SIZE, _default_char_map, ATLAS_SIZE
from minecraft_ocr.renderer import TextSpan, render_text
from minecraft_ocr.pipeline import (
    OCRResult,
    LineResult,
    CharResult,
    ocr,
    _estimate_background,
    _downsample,
    _extract_foreground,
    _detect_lines,
    _group_templates_by_width,
    _best_match,
    _dominant_color,
    _recognize_line,
    _detect_scale,
    _SPACE_ADVANCE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def atlas() -> FontAtlas:
    return FontAtlas.from_builtin()


@pytest.fixture(scope="module")
def templates_by_width(atlas) -> dict:
    return _group_templates_by_width(atlas)


# ---------------------------------------------------------------------------
# _estimate_background
# ---------------------------------------------------------------------------

class TestEstimateBackground:
    def test_uniform_black_image(self):
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        assert _estimate_background(img) == (0, 0, 0)

    def test_uniform_white_image(self):
        img = np.full((20, 20, 3), 255, dtype=np.uint8)
        assert _estimate_background(img) == (255, 255, 255)

    def test_corners_determine_result(self):
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        # Set all four corners to red
        img[0,  0]  = (200, 0, 0)
        img[0,  -1] = (200, 0, 0)
        img[-1, 0]  = (200, 0, 0)
        img[-1, -1] = (200, 0, 0)
        r, g, b = _estimate_background(img)
        assert r == 200 and g == 0 and b == 0

    def test_mixed_corners_returns_median(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[0,  0]  = (100, 100, 100)
        img[0,  -1] = (100, 100, 100)
        img[-1, 0]  = (200, 200, 200)
        img[-1, -1] = (200, 200, 200)
        result = _estimate_background(img)
        # Median of [100, 100, 200, 200] = 150
        assert result == (150, 150, 150)


# ---------------------------------------------------------------------------
# _downsample
# ---------------------------------------------------------------------------

class TestDownsample:
    def test_scale_1_returns_same_array(self):
        img = np.arange(75, dtype=np.uint8).reshape(5, 5, 3)
        out = _downsample(img, 1)
        np.testing.assert_array_equal(out, img)

    def test_scale_2_halves_dimensions(self):
        img = np.zeros((8, 10, 3), dtype=np.uint8)
        out = _downsample(img, 2)
        assert out.shape == (4, 5, 3)

    def test_scale_2_picks_correct_pixels(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        img[0, 0] = (255, 0, 0)
        img[0, 2] = (0, 255, 0)
        img[2, 0] = (0, 0, 255)
        out = _downsample(img, 2)
        assert tuple(out[0, 0]) == (255, 0, 0)
        assert tuple(out[0, 1]) == (0, 255, 0)
        assert tuple(out[1, 0]) == (0, 0, 255)

    def test_scale_4_reduces_by_4(self):
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        out = _downsample(img, 4)
        assert out.shape == (4, 4, 3)


# ---------------------------------------------------------------------------
# _extract_foreground
# ---------------------------------------------------------------------------

class TestExtractForeground:
    def test_plain_threshold_no_shadow(self):
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        img[2, 2] = (255, 255, 255)
        fg = _extract_foreground(img, (0, 0, 0), shadow=False)
        assert fg[2, 2]
        assert not fg[0, 0]

    def test_shadow_pixels_excluded(self):
        # Text pixel at (2,2) = white; shadow at (3,3) = (63,63,63) = 255>>2
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        img[2, 2] = (255, 255, 255)
        img[3, 3] = (63, 63, 63)
        fg = _extract_foreground(img, (0, 0, 0), shadow=True)
        assert fg[2, 2]      # text pixel kept
        assert not fg[3, 3]  # shadow pixel removed

    def test_background_pixels_excluded(self):
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        fg = _extract_foreground(img, (0, 0, 0), shadow=False)
        assert not fg.any()

    def test_non_shadow_non_bg_pixel_kept(self):
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        img[2, 2] = (255, 0, 0)  # red text
        img[1, 1] = (100, 50, 50)  # something unrelated nearby
        fg = _extract_foreground(img, (0, 0, 0), shadow=True)
        assert fg[2, 2]  # red pixel kept

    def test_shadow_of_dark_text_excluded(self):
        # §8 dark gray (85,85,85); shadow = (21,21,21)
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        img[2, 2] = (85, 85, 85)
        img[3, 3] = (21, 21, 21)
        fg = _extract_foreground(img, (0, 0, 0), shadow=True)
        assert fg[2, 2]
        assert not fg[3, 3]


# ---------------------------------------------------------------------------
# _detect_lines
# ---------------------------------------------------------------------------

class TestDetectLines:
    def test_no_foreground_returns_empty(self):
        fg = np.zeros((20, 20), dtype=bool)
        assert _detect_lines(fg) == []

    def test_single_line_at_top(self):
        fg = np.zeros((20, 20), dtype=bool)
        fg[0:8, 5:10] = True
        lines = _detect_lines(fg)
        assert lines == [0]

    def test_single_line_offset(self):
        fg = np.zeros((20, 20), dtype=bool)
        fg[4:12, 5:10] = True
        lines = _detect_lines(fg)
        assert lines == [4]

    def test_two_lines_separated(self):
        fg = np.zeros((25, 20), dtype=bool)
        fg[0:8,   5:10] = True
        fg[10:18, 5:10] = True
        lines = _detect_lines(fg)
        assert lines == [0, 10]

    def test_skips_cell_size_after_line_start(self):
        # Line starts at row 0; rows 0-7 are occupied.
        # If CELL_SIZE=8, next scan starts at row 8 — should not double-count.
        fg = np.zeros((10, 5), dtype=bool)
        fg[0:8, :] = True
        lines = _detect_lines(fg)
        assert len(lines) == 1
        assert lines[0] == 0


# ---------------------------------------------------------------------------
# _group_templates_by_width
# ---------------------------------------------------------------------------

class TestGroupTemplatesByWidth:
    def test_all_templates_accounted_for(self, atlas, templates_by_width):
        total = sum(len(v) for v in templates_by_width.values())
        assert total == len(atlas.all_templates())

    def test_widths_are_positive(self, templates_by_width):
        for w in templates_by_width:
            assert w > 0

    def test_widths_are_advance_minus_one(self, atlas, templates_by_width):
        for w, tmpls in templates_by_width.items():
            for t in tmpls:
                assert t.advance - 1 == w


# ---------------------------------------------------------------------------
# _best_match
# ---------------------------------------------------------------------------

class TestBestMatch:
    def test_perfect_match_is_confidence_1(self, atlas):
        tmpl = atlas.get("A")
        match = _best_match(tmpl.bitmap, [tmpl])
        assert match is not None
        t, conf = match
        assert t.char == "A"
        assert conf == pytest.approx(1.0)

    def test_all_wrong_gives_lower_confidence(self, atlas):
        tmpl_a = atlas.get("A")
        tmpl_b = atlas.get("B")
        # Match A's bitmap against B's template only
        if tmpl_a.bitmap.shape != tmpl_b.bitmap.shape:
            pytest.skip("A and B have different widths")
        match = _best_match(tmpl_a.bitmap, [tmpl_b])
        assert match is not None
        _, conf = match
        assert conf < 1.0

    def test_returns_none_for_empty_template_list(self, atlas):
        tmpl = atlas.get("A")
        assert _best_match(tmpl.bitmap, []) is None

    def test_shape_mismatch_template_skipped(self, atlas):
        tmpl_i = atlas.get("i")   # narrow (advance 2, bitmap width 1)
        tmpl_a = atlas.get("A")   # wide  (advance 6, bitmap width 5)
        # Try to match i's bitmap against A's template — shapes differ, should be skipped.
        match = _best_match(tmpl_i.bitmap, [tmpl_a])
        assert match is None

    def test_best_of_two_templates_selected(self, atlas):
        tmpl_a = atlas.get("A")
        tmpl_b = atlas.get("B")
        if tmpl_a.bitmap.shape != tmpl_b.bitmap.shape:
            pytest.skip("A and B have different widths")
        # Perfect match for A should beat B match when A's bitmap supplied.
        match = _best_match(tmpl_a.bitmap, [tmpl_a, tmpl_b])
        assert match is not None
        tmpl, _ = match
        assert tmpl.char == "A"


# ---------------------------------------------------------------------------
# _dominant_color
# ---------------------------------------------------------------------------

class TestDominantColor:
    def test_all_same_color(self):
        region = np.full((8, 5, 3), [200, 100, 50], dtype=np.uint8)
        mask = np.ones((8, 5), dtype=bool)
        assert _dominant_color(region, mask) == (200, 100, 50)

    def test_empty_mask_returns_white(self):
        region = np.zeros((8, 5, 3), dtype=np.uint8)
        mask = np.zeros((8, 5), dtype=bool)
        assert _dominant_color(region, mask) == (255, 255, 255)

    def test_median_of_mixed_colors(self):
        region = np.zeros((2, 2, 3), dtype=np.uint8)
        region[0, 0] = (100, 100, 100)
        region[0, 1] = (200, 200, 200)
        region[1, 0] = (100, 100, 100)
        region[1, 1] = (200, 200, 200)
        mask = np.ones((2, 2), dtype=bool)
        r, g, b = _dominant_color(region, mask)
        # Median of [100, 100, 200, 200] = 150
        assert r == 150 and g == 150 and b == 150


# ---------------------------------------------------------------------------
# Round-trip: render → OCR → assert text
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """The core Phase 2 validation: synthetic render → OCR → matches original."""

    @pytest.mark.parametrize("scale", [1, 2, 3, 4])
    def test_single_char_A(self, atlas, scale):
        img = render_text([TextSpan("A")], atlas, scale=scale, shadow=True)
        result = ocr(img, atlas, scale=scale)
        assert result.text == "A"

    @pytest.mark.parametrize("scale", [1, 2, 3, 4])
    def test_hello(self, atlas, scale):
        img = render_text([TextSpan("Hello")], atlas, scale=scale, shadow=True)
        result = ocr(img, atlas, scale=scale)
        assert result.text == "Hello"

    @pytest.mark.parametrize("scale", [1, 2, 3, 4])
    def test_hello_world_with_space(self, atlas, scale):
        img = render_text([TextSpan("Hello World")], atlas, scale=scale, shadow=True)
        result = ocr(img, atlas, scale=scale)
        assert result.text == "Hello World"

    def test_all_lowercase(self, atlas):
        text = "abcdefghijklmnopqrstuvwxyz"
        img = render_text([TextSpan(text)], atlas, scale=2, shadow=True)
        result = ocr(img, atlas, scale=2)
        assert result.text == text

    def test_all_uppercase(self, atlas):
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        img = render_text([TextSpan(text)], atlas, scale=2, shadow=True)
        result = ocr(img, atlas, scale=2)
        assert result.text == text

    def test_all_digits(self, atlas):
        text = "0123456789"
        img = render_text([TextSpan(text)], atlas, scale=2, shadow=True)
        result = ocr(img, atlas, scale=2)
        assert result.text == text

    @pytest.mark.parametrize("text", [
        "Hello, World!",
        "Score: 9999",
        "Player has joined the game",
        "<Player> test message",
    ])
    def test_common_chat_strings(self, atlas, text):
        img = render_text([TextSpan(text)], atlas, scale=2, shadow=True)
        result = ocr(img, atlas, scale=2)
        assert result.text == text

    def test_no_shadow_mode(self, atlas):
        img = render_text([TextSpan("Hello")], atlas, scale=2, shadow=False)
        result = ocr(img, atlas, scale=2, shadow=False)
        assert result.text == "Hello"

    def test_coloured_text_recognised_correctly(self, atlas):
        # Colour should not affect character recognition — only text shape matters.
        for color in [(255, 85, 85), (85, 255, 85), (85, 85, 255), (255, 170, 0)]:
            img = render_text([TextSpan("Hello", color=color)], atlas, scale=2, shadow=True)
            result = ocr(img, atlas, scale=2)
            assert result.text == "Hello", f"Failed for color {color}"

    def test_colour_detected_correctly(self, atlas):
        color = (255, 85, 85)  # §c Red
        img = render_text([TextSpan("A", color=color)], atlas, scale=2, shadow=True)
        result = ocr(img, atlas, scale=2)
        assert result.text == "A"
        r, g, b = result.lines[0].chars[0].color
        assert abs(r - 255) <= 10
        assert abs(g - 85)  <= 10
        assert abs(b - 85)  <= 10

    def test_confidence_is_high_for_clean_render(self, atlas):
        img = render_text([TextSpan("Hello")], atlas, scale=2, shadow=True)
        result = ocr(img, atlas, scale=2)
        assert result.mean_confidence >= 0.85

    def test_perfect_confidence_for_exact_template(self, atlas):
        # Single character rendered cleanly should match its template perfectly.
        img = render_text([TextSpan("A")], atlas, scale=1, shadow=False)
        result = ocr(img, atlas, scale=1, shadow=False)
        assert result.lines[0].chars[0].confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Scale auto-detection
# ---------------------------------------------------------------------------

class TestScaleDetection:
    @pytest.mark.parametrize("scale", [1, 2, 3, 4])
    def test_detects_correct_scale(self, atlas, scale):
        img = render_text([TextSpan("Hello World")], atlas, scale=scale, shadow=True)
        result = ocr(img, atlas, scale=None)  # auto-detect
        assert result.scale == scale

    def test_auto_detect_still_reads_correctly(self, atlas):
        img = render_text([TextSpan("Hello")], atlas, scale=2, shadow=True)
        result = ocr(img, atlas, scale=None)
        assert result.text == "Hello"


# ---------------------------------------------------------------------------
# Multi-line detection
# ---------------------------------------------------------------------------

class TestMultiLine:
    def _render_two_lines(self, atlas, line1: str, line2: str, scale: int = 2) -> Image.Image:
        """Stack two rendered lines with a 2px gap."""
        img1 = render_text([TextSpan(line1)], atlas, scale=scale, shadow=True)
        img2 = render_text([TextSpan(line2)], atlas, scale=scale, shadow=True)
        gap = 2
        total_h = img1.height + gap + img2.height
        max_w = max(img1.width, img2.width)
        canvas = Image.new("RGB", (max_w, total_h), (0, 0, 0))
        canvas.paste(img1, (0, 0))
        canvas.paste(img2, (0, img1.height + gap))
        return canvas

    def test_two_lines_detected(self, atlas):
        img = self._render_two_lines(atlas, "Hello", "World")
        result = ocr(img, atlas, scale=2)
        assert len(result.lines) == 2

    def test_two_lines_text_correct(self, atlas):
        img = self._render_two_lines(atlas, "Hello", "World")
        result = ocr(img, atlas, scale=2)
        assert result.lines[0].text == "Hello"
        assert result.lines[1].text == "World"

    def test_text_property_joins_with_newline(self, atlas):
        img = self._render_two_lines(atlas, "Hello", "World")
        result = ocr(img, atlas, scale=2)
        assert result.text == "Hello\nWorld"

    def test_line_y_coordinates_are_ordered(self, atlas):
        img = self._render_two_lines(atlas, "Hello", "World")
        result = ocr(img, atlas, scale=2)
        assert result.lines[0].y < result.lines[1].y


# ---------------------------------------------------------------------------
# OCRResult / LineResult properties
# ---------------------------------------------------------------------------

class TestResultProperties:
    def test_empty_result_text_is_empty(self):
        result = OCRResult()
        assert result.text == ""

    def test_empty_result_confidence_is_zero(self):
        result = OCRResult()
        assert result.mean_confidence == 0.0

    def test_single_line_text(self):
        result = OCRResult(lines=[
            LineResult(chars=[
                CharResult("H", 0.9, (255,255,255), 0),
                CharResult("i", 0.95, (255,255,255), 6),
            ])
        ], scale=2)
        assert result.text == "Hi"

    def test_line_result_mean_confidence(self):
        line = LineResult(chars=[
            CharResult("A", 0.8, (255,255,255), 0),
            CharResult("B", 1.0, (255,255,255), 6),
        ])
        assert line.mean_confidence == pytest.approx(0.9)

    def test_empty_line_confidence_is_zero(self):
        assert LineResult().mean_confidence == 0.0
