"""
Font atlas extraction, template library, and character width calculation.

The Minecraft default font is stored as a 128×128 PNG sprite sheet inside the
game JAR at assets/minecraft/textures/font/ascii.png.  It is laid out as a
16×16 grid of 8×8 character cells.  Each character occupies one cell; its
*actual* pixel width is determined by scanning the rightmost non-transparent
column, and its advance width is that pixel width plus one mandatory
inter-character gap pixel.

This module:
  - Loads the atlas from a JAR file or from a standalone PNG.
  - Builds a CharTemplate for every glyph: stores the raw bitmap plus
    precomputed bold and italic variants.
  - Exposes helpers used by the renderer and the OCR pipeline.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CELL_SIZE: int = 8    # each glyph cell is 8×8 font-pixels
GRID_SIZE: int = 16   # 16 columns × 16 rows = 256 cells
ATLAS_SIZE: int = 128  # expected atlas image dimension (px)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CharTemplate:
    """All pre-computed information for a single character glyph.

    Attributes:
        char:    The Unicode character this template represents.
        advance: Total advance width in font-pixels: glyph pixel width + 1 gap.
        bitmap:  Boolean array (8, advance-1) — the glyph's own pixels, no gap.
        bold:    Boolean array (8, advance)   — bold variant (1 px wider).
        italic:  Boolean array (8, advance-1) — italic variant (sheared in-place).
    """
    char: str
    advance: int
    bitmap: np.ndarray
    bold: np.ndarray
    italic: np.ndarray


# ---------------------------------------------------------------------------
# FontAtlas
# ---------------------------------------------------------------------------

class FontAtlas:
    """Parsed font atlas with pre-built per-character templates.

    Typical usage::

        atlas = FontAtlas.from_jar("minecraft-1.21.jar")
        tmpl = atlas.get("A")
        print(tmpl.advance)  # 6  (5 glyph pixels + 1 gap)
    """

    def __init__(self, atlas_image: Image.Image, char_map: list[str]) -> None:
        """
        Args:
            atlas_image: 128×128 RGBA PIL image (the ascii.png sprite sheet).
            char_map:    256-entry list mapping cell index → Unicode character.
        """
        if atlas_image.size != (ATLAS_SIZE, ATLAS_SIZE):
            raise ValueError(
                f"Expected {ATLAS_SIZE}×{ATLAS_SIZE} atlas, got {atlas_image.size}. "
                "Custom resource-pack atlases are not supported."
            )
        # Store alpha as a (128, 128) bool array — we only care about opacity.
        rgba = np.array(atlas_image.convert("RGBA"))
        self._alpha: np.ndarray = rgba[:, :, 3] > 0  # (128, 128) bool
        self.char_map: list[str] = char_map
        self.templates: dict[str, CharTemplate] = {}
        self._build_all()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_jar(cls, jar_path: str | Path) -> FontAtlas:
        """Load the atlas directly from a Minecraft JAR (which is a ZIP)."""
        jar_path = Path(jar_path)
        with zipfile.ZipFile(jar_path) as jar:
            with jar.open("assets/minecraft/textures/font/ascii.png") as f:
                img = Image.open(io.BytesIO(f.read()))
                img.load()  # ensure data is read before the ZipFile closes
            try:
                with jar.open("assets/minecraft/font/default.json") as f:
                    font_def = json.load(f)
                char_map = _parse_char_map(font_def)
            except KeyError:
                # Older JAR layouts may not include default.json; fall back.
                char_map = _default_char_map()
        return cls(img, char_map)

    @classmethod
    def from_png(
        cls,
        png_path: str | Path,
        char_map: list[str] | None = None,
    ) -> FontAtlas:
        """Load from a standalone ascii.png (e.g. extracted from JAR manually)."""
        img = Image.open(Path(png_path))
        return cls(img, char_map if char_map is not None else _default_char_map())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, char: str) -> CharTemplate | None:
        """Return the template for *char*, or None if not in the atlas."""
        return self.templates.get(char)

    def all_templates(self) -> list[CharTemplate]:
        """Return all built templates (order is not guaranteed)."""
        return list(self.templates.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cell_alpha(self, cell_index: int) -> np.ndarray:
        """Return the 8×8 boolean alpha mask for the given cell index."""
        col = cell_index % GRID_SIZE
        row = cell_index // GRID_SIZE
        x0 = col * CELL_SIZE
        y0 = row * CELL_SIZE
        return self._alpha[y0:y0 + CELL_SIZE, x0:x0 + CELL_SIZE]  # (8,8) bool

    def _build_all(self) -> None:
        for cell_index in range(GRID_SIZE * GRID_SIZE):
            if cell_index >= len(self.char_map):
                break
            ch = self.char_map[cell_index]
            if not ch or ch == "\x00":
                continue
            alpha = self._cell_alpha(cell_index)
            if not alpha.any():
                # Fully transparent cell — no glyph defined here.
                continue
            advance = _compute_advance(alpha)
            char_w = advance - 1  # pixel width without the gap column
            bitmap = alpha[:, :char_w].copy()
            self.templates[ch] = CharTemplate(
                char=ch,
                advance=advance,
                bitmap=bitmap,
                bold=_make_bold(bitmap),
                italic=_make_italic(bitmap),
            )


# ---------------------------------------------------------------------------
# Module-level helpers (also used by tests and the renderer)
# ---------------------------------------------------------------------------

def _compute_advance(cell_alpha: np.ndarray) -> int:
    """Compute the advance width of a glyph from its 8×8 alpha mask.

    Scans right-to-left for the first non-transparent column.
    Returns that column index + 1 (for the pixel itself) + 1 (for the gap).
    Falls back to 4 (the width of a space) for fully transparent cells.
    """
    for x in range(CELL_SIZE - 1, -1, -1):
        if cell_alpha[:, x].any():
            return x + 2  # +1 for 0-index→width, +1 for inter-char gap
    return 4  # space / empty cell default


def _make_bold(bitmap: np.ndarray) -> np.ndarray:
    """Compute the bold variant of a glyph bitmap.

    Bold is implemented by ORing the bitmap with itself shifted 1 px right,
    producing a result that is 1 column wider than the input.

    Args:
        bitmap: Boolean array (8, W).

    Returns:
        Boolean array (8, W+1).
    """
    h, w = bitmap.shape
    out = np.zeros((h, w + 1), dtype=bool)
    out[:, :w] |= bitmap   # original pixels
    out[:, 1:] |= bitmap   # shifted-right pixels
    return out


def _make_italic(bitmap: np.ndarray) -> np.ndarray:
    """Compute the italic variant of a glyph bitmap.

    Minecraft's italic shear shifts each row right by ``(CELL_SIZE - 1 - row) // 2``
    pixels, where row 0 is the top and row 7 is the bottom (unshifted baseline).

    Row shifts:
        row 0 → +3 px     row 1 → +3 px
        row 2 → +2 px     row 3 → +2 px
        row 4 → +1 px     row 5 → +1 px
        row 6 → +0 px     row 7 → +0 px

    The output is the same shape as the input; pixels shifted beyond the right
    edge are cropped (they bleed into the next character's space in practice, but
    for template matching purposes this simplification is acceptable in Phase 1-2).

    Args:
        bitmap: Boolean array (8, W).

    Returns:
        Boolean array (8, W) — same shape, pixels sheared in-place.
    """
    h, w = bitmap.shape
    out = np.zeros((h, w), dtype=bool)
    for row in range(h):
        shift = (CELL_SIZE - 1 - row) // 2
        src_cols = w - shift  # how many source columns still fit after shift
        if src_cols > 0:
            out[row, shift:shift + src_cols] = bitmap[row, :src_cols]
    return out


def _default_char_map() -> list[str]:
    """Default char map: cell index == Unicode code point (covers ASCII/Latin-1)."""
    return [chr(i) for i in range(GRID_SIZE * GRID_SIZE)]


def _parse_char_map(font_def: dict) -> list[str]:
    """Extract the 256-entry char map from a parsed default.json font definition.

    default.json contains a list of providers.  We look for the first bitmap
    provider whose file path references ascii.png and read its ``chars`` rows.
    Cells not covered by the provider retain their default (chr(index)) mapping.
    """
    char_map = _default_char_map()
    for provider in font_def.get("providers", []):
        if provider.get("type") != "bitmap":
            continue
        if "ascii.png" not in provider.get("file", ""):
            continue
        for row_idx, row_str in enumerate(provider.get("chars", [])):
            for col_idx, ch in enumerate(row_str):
                idx = row_idx * GRID_SIZE + col_idx
                if idx < len(char_map):
                    char_map[idx] = ch
        break  # only the first matching provider
    return char_map
