"""Tests for minecraft_ocr.atlas"""

import numpy as np
import pytest
from PIL import Image

from minecraft_ocr.atlas import (
    CELL_SIZE,
    GRID_SIZE,
    ATLAS_SIZE,
    CharTemplate,
    FontAtlas,
    _compute_advance,
    _default_char_map,
    _make_bold,
    _make_italic,
    _parse_char_map,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_atlas_image() -> Image.Image:
    """128×128 fully transparent atlas."""
    return Image.new("RGBA", (ATLAS_SIZE, ATLAS_SIZE), (0, 0, 0, 0))


def _atlas_with_char(char: str, cell_pixel_cols: int) -> FontAtlas:
    """Build a minimal FontAtlas with one character drawn *cell_pixel_cols* wide."""
    img = _blank_atlas_image()
    code = ord(char)
    col = code % GRID_SIZE
    row = code // GRID_SIZE
    x0 = col * CELL_SIZE
    y0 = row * CELL_SIZE
    pixels = img.load()
    for fy in range(CELL_SIZE):
        for fx in range(cell_pixel_cols):
            pixels[x0 + fx, y0 + fy] = (255, 255, 255, 255)
    return FontAtlas(img, _default_char_map())


# ---------------------------------------------------------------------------
# _compute_advance
# ---------------------------------------------------------------------------

class TestComputeAdvance:
    def test_empty_cell_returns_space_width(self):
        alpha = np.zeros((CELL_SIZE, CELL_SIZE), dtype=bool)
        assert _compute_advance(alpha) == 4

    def test_single_pixel_at_col_zero(self):
        alpha = np.zeros((CELL_SIZE, CELL_SIZE), dtype=bool)
        alpha[4, 0] = True
        # rightmost non-empty col = 0, advance = 0 + 2 = 2
        assert _compute_advance(alpha) == 2

    def test_single_pixel_at_col_two(self):
        alpha = np.zeros((CELL_SIZE, CELL_SIZE), dtype=bool)
        alpha[4, 2] = True
        assert _compute_advance(alpha) == 4  # 2 + 2

    def test_full_width_cell(self):
        alpha = np.ones((CELL_SIZE, CELL_SIZE), dtype=bool)
        # rightmost non-empty col = 7, advance = 7 + 2 = 9
        assert _compute_advance(alpha) == 9

    def test_standard_5px_char(self):
        alpha = np.zeros((CELL_SIZE, CELL_SIZE), dtype=bool)
        alpha[:, :5] = True  # columns 0-4 set
        assert _compute_advance(alpha) == 6  # 4 + 2


# ---------------------------------------------------------------------------
# _make_bold
# ---------------------------------------------------------------------------

class TestMakeBold:
    def test_output_is_one_column_wider(self):
        bitmap = np.ones((CELL_SIZE, 5), dtype=bool)
        bold = _make_bold(bitmap)
        assert bold.shape == (CELL_SIZE, 6)

    def test_no_pixels_lost_from_original(self):
        bitmap = np.eye(CELL_SIZE, 5, dtype=bool)
        bold = _make_bold(bitmap)
        # Every pixel that was True in original should still be True in bold
        assert (bold[:, :5] | bold[:, 1:6])[bitmap].all()

    def test_all_true_input_stays_all_true(self):
        bitmap = np.ones((CELL_SIZE, 5), dtype=bool)
        bold = _make_bold(bitmap)
        assert bold.all()

    def test_single_pixel_duplicated_right(self):
        bitmap = np.zeros((CELL_SIZE, 5), dtype=bool)
        bitmap[3, 2] = True
        bold = _make_bold(bitmap)
        assert bold[3, 2]   # original position
        assert bold[3, 3]   # shifted right by 1
        # No other pixels should be set
        assert bold.sum() == 2

    def test_shape_for_minimal_bitmap(self):
        bitmap = np.zeros((CELL_SIZE, 1), dtype=bool)
        bold = _make_bold(bitmap)
        assert bold.shape == (CELL_SIZE, 2)


# ---------------------------------------------------------------------------
# _make_italic
# ---------------------------------------------------------------------------

class TestMakeItalic:
    def test_output_same_shape_as_input(self):
        bitmap = np.ones((CELL_SIZE, 5), dtype=bool)
        italic = _make_italic(bitmap)
        assert italic.shape == bitmap.shape

    def test_bottom_two_rows_unshifted(self):
        """Rows 6 and 7 have shift = 0."""
        bitmap = np.ones((CELL_SIZE, 5), dtype=bool)
        italic = _make_italic(bitmap)
        np.testing.assert_array_equal(italic[6], bitmap[6])
        np.testing.assert_array_equal(italic[7], bitmap[7])

    def test_top_row_shifted_right_by_three(self):
        """Row 0: shift = (8-1-0)//2 = 3."""
        bitmap = np.zeros((CELL_SIZE, 5), dtype=bool)
        bitmap[0, 0] = True  # top-left pixel
        italic = _make_italic(bitmap)
        assert not italic[0, 0]    # original position cleared
        assert italic[0, 3]        # shifted to col 3

    def test_row_two_shifted_right_by_two(self):
        """Row 2: shift = (8-1-2)//2 = 2."""
        bitmap = np.zeros((CELL_SIZE, 5), dtype=bool)
        bitmap[2, 0] = True
        italic = _make_italic(bitmap)
        assert not italic[2, 0]
        assert italic[2, 2]

    def test_row_four_shifted_right_by_one(self):
        """Row 4: shift = (8-1-4)//2 = 1."""
        bitmap = np.zeros((CELL_SIZE, 5), dtype=bool)
        bitmap[4, 0] = True
        italic = _make_italic(bitmap)
        assert not italic[4, 0]
        assert italic[4, 1]

    def test_overflow_is_cropped(self):
        """Pixels shifted beyond the right edge are just lost — no shape change."""
        bitmap = np.zeros((CELL_SIZE, 2), dtype=bool)
        bitmap[0, :] = True  # row 0 shifts +3, but bitmap is only 2 wide
        italic = _make_italic(bitmap)
        assert italic.shape == (CELL_SIZE, 2)
        # Row 0, cols 0 and 1 both shift to cols 3 and 4, both out of bounds
        assert not italic[0].any()


# ---------------------------------------------------------------------------
# FontAtlas construction & validation
# ---------------------------------------------------------------------------

class TestFontAtlasConstruction:
    def test_rejects_wrong_size(self):
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        with pytest.raises(ValueError, match="128"):
            FontAtlas(img, _default_char_map())

    def test_builds_template_for_printable_ascii(self):
        atlas = _atlas_with_char("A", 5)
        tmpl = atlas.get("A")
        assert tmpl is not None
        assert tmpl.char == "A"

    def test_advance_matches_pixel_width_plus_gap(self):
        atlas = _atlas_with_char("B", 4)
        tmpl = atlas.get("B")
        assert tmpl.advance == 5  # 4 pixels + 1 gap

    def test_bitmap_shape_is_8_by_char_width(self):
        atlas = _atlas_with_char("C", 5)
        tmpl = atlas.get("C")
        assert tmpl.bitmap.shape == (CELL_SIZE, 5)

    def test_bold_shape_is_one_wider_than_bitmap(self):
        atlas = _atlas_with_char("D", 5)
        tmpl = atlas.get("D")
        assert tmpl.bold.shape == (CELL_SIZE, tmpl.bitmap.shape[1] + 1)

    def test_italic_shape_equals_bitmap_shape(self):
        atlas = _atlas_with_char("E", 5)
        tmpl = atlas.get("E")
        assert tmpl.italic.shape == tmpl.bitmap.shape

    def test_unknown_char_returns_none(self):
        atlas = FontAtlas(_blank_atlas_image(), _default_char_map())
        assert atlas.get("A") is None  # no pixels drawn, treated as space/empty

    def test_all_templates_not_empty_with_drawn_char(self):
        atlas = _atlas_with_char("X", 5)
        # Blank cells are skipped; 'X' should be in all_templates
        chars = {t.char for t in atlas.all_templates()}
        assert "X" in chars


# ---------------------------------------------------------------------------
# _parse_char_map
# ---------------------------------------------------------------------------

class TestParseCharMap:
    def test_default_map_is_identity(self):
        char_map = _default_char_map()
        assert char_map[65] == "A"
        assert char_map[48] == "0"
        assert len(char_map) == GRID_SIZE * GRID_SIZE

    def test_parse_overrides_entries(self):
        font_def = {
            "providers": [{
                "type": "bitmap",
                "file": "minecraft:font/ascii.png",
                "chars": [
                    "\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000",
                    *["\u0000" * GRID_SIZE] * (GRID_SIZE - 1),
                ]
            }]
        }
        # row 0 is all null — cell 0 should remain chr(0) since we set it to \0
        char_map = _parse_char_map(font_def)
        assert len(char_map) == GRID_SIZE * GRID_SIZE

    def test_non_bitmap_provider_ignored(self):
        font_def = {
            "providers": [{
                "type": "ttf",
                "file": "something.ttf",
                "chars": ["ABCDEFGHIJKLMNOP"],
            }]
        }
        char_map = _parse_char_map(font_def)
        # Should be unchanged from default
        assert char_map[65] == "A"

    def test_non_ascii_provider_file_ignored(self):
        font_def = {
            "providers": [{
                "type": "bitmap",
                "file": "minecraft:font/accented.png",
                "chars": ["AAAAAAAAAAAAAAAA"],
            }]
        }
        char_map = _parse_char_map(font_def)
        # Should be unchanged — file doesn't contain ascii.png
        assert char_map[65] == "A"
