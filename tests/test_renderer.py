"""Tests for minecraft_ocr.renderer"""

import numpy as np
import pytest
from PIL import Image

from minecraft_ocr.atlas import FontAtlas, _default_char_map, CELL_SIZE, ATLAS_SIZE
from minecraft_ocr.renderer import TextSpan, render_text, measure_width


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_atlas(char: str = "A", pixel_width: int = 5) -> FontAtlas:
    """Minimal atlas with one solid rectangle for *char*."""
    img = Image.new("RGBA", (ATLAS_SIZE, ATLAS_SIZE), (0, 0, 0, 0))
    code = ord(char)
    col = code % 16
    row = code // 16
    x0 = col * CELL_SIZE
    y0 = row * CELL_SIZE
    pixels = img.load()
    for fy in range(CELL_SIZE):
        for fx in range(pixel_width):
            pixels[x0 + fx, y0 + fy] = (255, 255, 255, 255)
    return FontAtlas(img, _default_char_map())


WHITE = (255, 255, 255)
RED   = (255,   0,   0)
BLACK = (0,     0,   0)


# ---------------------------------------------------------------------------
# render_text — basic sanity
# ---------------------------------------------------------------------------

class TestRenderBasics:
    def test_returns_pil_image(self):
        atlas = _make_atlas()
        img = render_text([TextSpan("A")], atlas, scale=1, shadow=False)
        assert isinstance(img, Image.Image)

    def test_mode_is_rgb(self):
        atlas = _make_atlas()
        img = render_text([TextSpan("A")], atlas, scale=1, shadow=False)
        assert img.mode == "RGB"

    def test_invalid_scale_too_low(self):
        atlas = _make_atlas()
        with pytest.raises(ValueError):
            render_text([TextSpan("A")], atlas, scale=0)

    def test_invalid_scale_too_high(self):
        atlas = _make_atlas()
        with pytest.raises(ValueError):
            render_text([TextSpan("A")], atlas, scale=5)

    def test_empty_spans_produces_tiny_image(self):
        atlas = _make_atlas()
        img = render_text([], atlas, scale=1, shadow=False)
        # No text → width = 0, height = 8
        assert img.width == 0
        assert img.height == CELL_SIZE


# ---------------------------------------------------------------------------
# render_text — dimensions
# ---------------------------------------------------------------------------

class TestRenderDimensions:
    def test_height_is_cell_size_times_scale_no_shadow(self):
        atlas = _make_atlas()
        for scale in (1, 2, 3, 4):
            img = render_text([TextSpan("A")], atlas, scale=scale, shadow=False)
            assert img.height == CELL_SIZE * scale, f"scale={scale}"

    def test_shadow_adds_one_scale_unit_to_both_dimensions(self):
        atlas = _make_atlas()
        for scale in (1, 2):
            no_shadow = render_text([TextSpan("A")], atlas, scale=scale, shadow=False)
            with_shadow = render_text([TextSpan("A")], atlas, scale=scale, shadow=True)
            assert with_shadow.width  == no_shadow.width  + scale
            assert with_shadow.height == no_shadow.height + scale

    def test_scale_2_is_exactly_double_scale_1(self):
        atlas = _make_atlas()
        spans = [TextSpan("A")]
        img1 = render_text(spans, atlas, scale=1, shadow=False)
        img2 = render_text(spans, atlas, scale=2, shadow=False)
        assert img2.width  == img1.width  * 2
        assert img2.height == img1.height * 2

    def test_width_equals_measure_width_times_scale(self):
        atlas = _make_atlas()
        spans = [TextSpan("A")]
        w = measure_width(spans, atlas)
        img = render_text(spans, atlas, scale=2, shadow=False)
        assert img.width == w * 2


# ---------------------------------------------------------------------------
# render_text — pixel colour correctness
# ---------------------------------------------------------------------------

class TestRenderColours:
    def test_foreground_colour_appears_in_image(self):
        atlas = _make_atlas()
        img = render_text([TextSpan("A", color=RED)], atlas, scale=1, shadow=False, background=BLACK)
        arr = np.array(img)
        assert ((arr[:, :, 0] == 255) & (arr[:, :, 1] == 0) & (arr[:, :, 2] == 0)).any()

    def test_background_colour_is_present(self):
        atlas = _make_atlas()
        bg = (10, 20, 30)
        img = render_text([TextSpan("A", color=WHITE)], atlas, scale=1, shadow=False, background=bg)
        arr = np.array(img)
        assert ((arr[:, :, 0] == bg[0]) & (arr[:, :, 1] == bg[1]) & (arr[:, :, 2] == bg[2])).any()

    def test_shadow_colour_is_quarter_of_text_colour(self):
        atlas = _make_atlas()
        text_color = (200, 100, 60)
        expected_shadow = (200 >> 2, 100 >> 2, 60 >> 2)  # (50, 25, 15)
        img = render_text(
            [TextSpan("A", color=text_color)],
            atlas, scale=2, shadow=True, background=BLACK,
        )
        arr = np.array(img)
        shadow_mask = (
            (arr[:, :, 0] == expected_shadow[0]) &
            (arr[:, :, 1] == expected_shadow[1]) &
            (arr[:, :, 2] == expected_shadow[2])
        )
        assert shadow_mask.any(), "Expected shadow pixels not found"

    def test_text_overwrites_shadow_pixels(self):
        """At scale 1, shadow is at (+1,+1). The first column of text should
        NOT contain shadow colour — text pass overwrites where they overlap."""
        atlas = _make_atlas("A", pixel_width=5)
        text_color = (200, 100, 60)
        shadow_color = (200 >> 2, 100 >> 2, 60 >> 2)
        img = render_text(
            [TextSpan("A", color=text_color)],
            atlas, scale=1, shadow=True, background=BLACK,
        )
        arr = np.array(img)
        # Column 0 (leftmost) should have text colour, not shadow colour
        col0 = arr[:CELL_SIZE, 0]
        has_text   = ((col0[:, 0] == text_color[0]) & (col0[:, 1] == text_color[1])).any()
        shadow_only = ((col0[:, 0] == shadow_color[0]) & (col0[:, 1] == shadow_color[1])).all()
        assert has_text
        assert not shadow_only


# ---------------------------------------------------------------------------
# render_text — bold and italic
# ---------------------------------------------------------------------------

class TestRenderFormatting:
    def test_bold_image_wider_than_normal(self):
        atlas = _make_atlas()
        normal = render_text([TextSpan("A", bold=False)], atlas, scale=1, shadow=False)
        bold   = render_text([TextSpan("A", bold=True)],  atlas, scale=1, shadow=False)
        assert bold.width == normal.width + 1

    def test_italic_same_width_as_normal(self):
        atlas = _make_atlas()
        normal = render_text([TextSpan("A", italic=False)], atlas, scale=1, shadow=False)
        italic = render_text([TextSpan("A", italic=True)],  atlas, scale=1, shadow=False)
        assert italic.width == normal.width

    def test_underline_adds_pixels_below_cell(self):
        atlas = _make_atlas()
        img = render_text(
            [TextSpan("A", color=WHITE, underline=True)],
            atlas, scale=1, shadow=False, background=BLACK,
        )
        arr = np.array(img)
        # Row 8 (index CELL_SIZE) should contain text-colour pixels (the underline)
        assert img.height >= CELL_SIZE + 1, "Image too short to contain underline row"
        underline_row = arr[CELL_SIZE]
        assert (underline_row == WHITE).all(axis=-1).any()

    def test_strikethrough_adds_pixels_mid_cell(self):
        from minecraft_ocr.renderer import _STRIKETHROUGH_ROW
        atlas = _make_atlas()
        img = render_text(
            [TextSpan("A", color=RED, strikethrough=True)],
            atlas, scale=1, shadow=False, background=BLACK,
        )
        arr = np.array(img)
        st_row = arr[_STRIKETHROUGH_ROW]
        assert (st_row == RED).all(axis=-1).any()


# ---------------------------------------------------------------------------
# measure_width
# ---------------------------------------------------------------------------

class TestMeasureWidth:
    def test_single_char(self):
        atlas = _make_atlas("A", pixel_width=5)
        tmpl = atlas.get("A")
        w = measure_width([TextSpan("A")], atlas)
        assert w == tmpl.advance

    def test_two_chars(self):
        atlas = _make_atlas("A", pixel_width=5)
        tmpl = atlas.get("A")
        w = measure_width([TextSpan("AA")], atlas)
        assert w == tmpl.advance * 2

    def test_bold_adds_one_per_char(self):
        atlas = _make_atlas("A", pixel_width=5)
        tmpl = atlas.get("A")
        w_normal = measure_width([TextSpan("AA", bold=False)], atlas)
        w_bold   = measure_width([TextSpan("AA", bold=True)],  atlas)
        assert w_bold == w_normal + 2  # +1 per character

    def test_unknown_char_uses_fallback_width(self):
        atlas = _make_atlas("A", pixel_width=5)
        # '§' is unlikely to be drawn in our minimal atlas
        w = measure_width([TextSpan("§")], atlas)
        assert w == 4  # fallback space width

    def test_empty_span(self):
        atlas = _make_atlas()
        assert measure_width([TextSpan("")], atlas) == 0

    def test_multi_span(self):
        atlas = _make_atlas("A", pixel_width=5)
        tmpl = atlas.get("A")
        spans = [TextSpan("A"), TextSpan("A")]
        assert measure_width(spans, atlas) == tmpl.advance * 2
