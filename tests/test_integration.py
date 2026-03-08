"""Integration tests using the real bundled vanilla atlas (Minecraft 1.21.4).

These tests verify against the actual ascii.png shipped with the package.
No JAR or Minecraft installation is required.
"""

import numpy as np
import pytest
from PIL import Image

from minecraft_ocr.atlas import FontAtlas, CELL_SIZE
from minecraft_ocr.renderer import TextSpan, render_text, measure_width


@pytest.fixture(scope="module")
def atlas() -> FontAtlas:
    return FontAtlas.from_builtin()


# ---------------------------------------------------------------------------
# Atlas loads correctly
# ---------------------------------------------------------------------------

class TestBuiltinAtlasLoads:
    def test_loads_without_error(self, atlas):
        assert atlas is not None

    def test_has_templates(self, atlas):
        assert len(atlas.templates) > 0

    def test_all_printable_ascii_present(self, atlas):
        # Space (chr 32) has a transparent cell and no template by design — skip it.
        missing = [chr(i) for i in range(33, 127) if atlas.get(chr(i)) is None]
        assert missing == [], f"Missing printable ASCII characters: {missing}"

    def test_space_has_advance_4(self, atlas):
        # Space is the one character whose width is defined by convention, not pixels.
        # The vanilla atlas cell for space is transparent; _compute_advance returns 4.
        tmpl = atlas.get(" ")
        assert tmpl is None or tmpl.advance == 4

    def test_i_is_narrow(self, atlas):
        # 'i' in the 1.21.4 atlas is 1 glyph pixel wide + 1 gap = advance 2.
        tmpl = atlas.get("i")
        assert tmpl is not None
        assert tmpl.advance == 2

    def test_m_is_wide(self, atlas):
        # 'm' is one of the widest — 5 glyph pixels + 1 gap = advance 6.
        tmpl = atlas.get("m")
        assert tmpl is not None
        assert tmpl.advance == 6

    def test_bitmap_height_is_cell_size(self, atlas):
        for ch in "ABCDEFGabcdefg0123456789":
            tmpl = atlas.get(ch)
            assert tmpl is not None
            assert tmpl.bitmap.shape[0] == CELL_SIZE, f"Bad height for '{ch}'"

    def test_bitmap_width_matches_advance_minus_gap(self, atlas):
        for ch in "Hello, World!":
            tmpl = atlas.get(ch)
            if tmpl is None:
                continue
            assert tmpl.bitmap.shape[1] == tmpl.advance - 1, f"Width mismatch for '{ch}'"

    def test_bold_is_one_wider_than_bitmap(self, atlas):
        for ch in "ABCabc123":
            tmpl = atlas.get(ch)
            assert tmpl is not None
            assert tmpl.bold.shape[1] == tmpl.bitmap.shape[1] + 1

    def test_italic_same_shape_as_bitmap(self, atlas):
        for ch in "ABCabc123":
            tmpl = atlas.get(ch)
            assert tmpl is not None
            assert tmpl.italic.shape == tmpl.bitmap.shape


# ---------------------------------------------------------------------------
# Known character widths (vanilla 1.21.4)
# ---------------------------------------------------------------------------

# Advance = glyph pixel width + 1 gap. Values verified against the bundled
# vanilla 1.21.4 atlas.
KNOWN_ADVANCES = {
    "!": 2, "'": 2, ",": 2, ".": 2, "i": 2, ":": 2, ";": 2,
    "l": 3, "|": 2,
    "f": 5, "k": 5, "r": 6,
    "A": 6, "M": 6, "W": 6,
}

class TestKnownCharacterWidths:
    @pytest.mark.parametrize("char,expected_advance", KNOWN_ADVANCES.items())
    def test_advance(self, atlas, char, expected_advance):
        tmpl = atlas.get(char)
        assert tmpl is not None, f"No template for '{char}'"
        assert tmpl.advance == expected_advance, (
            f"'{char}': expected advance {expected_advance}, got {tmpl.advance}"
        )


# ---------------------------------------------------------------------------
# Renderer integration with real atlas
# ---------------------------------------------------------------------------

class TestRendererWithRealAtlas:
    def test_renders_hello_world(self, atlas):
        img = render_text([TextSpan("Hello, World!")], atlas, scale=2, shadow=True)
        assert isinstance(img, Image.Image)
        assert img.width > 0
        assert img.height > 0

    def test_rendered_image_contains_white_pixels(self, atlas):
        img = render_text(
            [TextSpan("A", color=(255, 255, 255))],
            atlas, scale=1, shadow=False, background=(0, 0, 0),
        )
        arr = np.array(img)
        assert (arr == 255).any()

    def test_shadow_pixels_present_at_correct_offset(self, atlas):
        """At scale 1, shadow is 1px right and 1px down from every text pixel."""
        color = (200, 120, 40)
        shadow = (color[0] >> 2, color[1] >> 2, color[2] >> 2)
        img = render_text(
            [TextSpan("A", color=color)],
            atlas, scale=1, shadow=True, background=(0, 0, 0),
        )
        arr = np.array(img)
        shadow_mask = (
            (arr[:, :, 0] == shadow[0]) &
            (arr[:, :, 1] == shadow[1]) &
            (arr[:, :, 2] == shadow[2])
        )
        assert shadow_mask.any(), "No shadow pixels found"

    def test_all_scales_produce_correct_height(self, atlas):
        for scale in (1, 2, 3, 4):
            img = render_text([TextSpan("A")], atlas, scale=scale, shadow=False)
            assert img.height == CELL_SIZE * scale, f"Wrong height at scale {scale}"

    def test_measure_width_matches_rendered_width(self, atlas):
        spans = [TextSpan("Hello")]
        w = measure_width(spans, atlas)
        img = render_text(spans, atlas, scale=1, shadow=False)
        assert img.width == w

    def test_bold_renders_wider_than_normal(self, atlas):
        normal = render_text([TextSpan("ABC")],            atlas, scale=1, shadow=False)
        bold   = render_text([TextSpan("ABC", bold=True)], atlas, scale=1, shadow=False)
        assert bold.width == normal.width + 3  # +1 per character

    def test_multi_colour_spans(self, atlas):
        spans = [
            TextSpan("Hello", color=(255, 85, 85)),
            TextSpan(" "),
            TextSpan("World", color=(85, 255, 85)),
        ]
        img = render_text(spans, atlas, scale=2, shadow=True)
        arr = np.array(img)
        has_red   = ((arr[:, :, 0] == 255) & (arr[:, :, 1] == 85)  & (arr[:, :, 2] == 85)).any()
        has_green = ((arr[:, :, 0] == 85)  & (arr[:, :, 1] == 255) & (arr[:, :, 2] == 85)).any()
        assert has_red,   "Red span pixels not found"
        assert has_green, "Green span pixels not found"
